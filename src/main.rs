#[macro_use] extern crate rocket;

use rocket::serde::{json::Json, Deserialize, Serialize};
use rocket::tokio::task;
use rocket::{Build, Rocket};
use rocket_cors::{AllowedOrigins, CorsOptions};
use dotenv::dotenv;
use std::env;
use postgres::{Client, NoTls};
use std::collections::HashMap;
use std::error::Error;
use reqwest::Client as ReqwestClient;


//
// Structures for API Communication
//

#[derive(Debug, Deserialize)]
#[serde(crate = "rocket::serde")]
struct NewInteraction {
    source: i32,
    target: i32,
    rating: f32,
}

#[derive(Debug, Serialize)]
#[serde(crate = "rocket::serde")]
struct InteractionResponse {
    status: String,
    is_anomaly: bool,
}

#[derive(Debug, Serialize)]
#[serde(crate = "rocket::serde")]
struct StatsResponse {
    total_interactions: u32,
    normal_interactions: u32,
    anomalous_interactions: u32,
    anomaly_ratio: f32,
}

#[derive(Debug, Serialize)]
#[serde(crate = "rocket::serde")]
struct InteractionRecord {
    source: i32,
    target: i32,
    rating: f32,
    timestamp: i64,
    anomaly: i16,
}

//
// Structures for the GNN Inference Service Request/Response
//

#[derive(Debug, Serialize)]
#[serde(crate = "rocket::serde")]
struct GnnRequest {
    source: i32,
    target: i32,
    rating: f32,
    // You can add extra features if your model needs them.
}

#[derive(Debug, Deserialize)]
#[serde(crate = "rocket::serde")]
struct GnnResponse {
    predicted_rating: f32,
    error: f32,
    is_anomaly: bool,
}

//
// Function to call the Python GNN Inference API asynchronously
//

async fn process_interaction_gnn(new_int: NewInteraction) -> Result<bool, Box<dyn Error + Send + Sync>> {
    // Build the payload according to what your Python model expects.
    let payload = GnnRequest {
        source: new_int.source,
        target: new_int.target,
        rating: new_int.rating,
    };

    // Get the GNN service URL from an environment variable,
    // or default to a given URL.
    let gnn_service_url = env::var("GNN_SERVICE_URL")
        .unwrap_or_else(|_| "http://localhost:8000/predict_interaction".into());

    // Create an asynchronous HTTP client.
    let client = ReqwestClient::new();
    let response = client
        .post(&gnn_service_url)
        .json(&payload)
        .send()
        .await?
        .error_for_status()?  // Convert HTTP errors to Rust errors.
        .json::<GnnResponse>()
        .await?;

    Ok(response.is_anomaly)
}

//
// API Endpoint: /add_interaction
// Calls the GNN service to classify the interaction as anomalous or not.
//

#[post("/add_interaction", format = "json", data = "<new_int>")]
async fn add_interaction(new_int: Json<NewInteraction>) -> Json<InteractionResponse> {
    let new_int_inner = new_int.into_inner();
    // Offload the blocking inference to a spawned async task.
    let result = task::spawn(async move {
        process_interaction_gnn(new_int_inner).await
    }).await;

    match result {
        Ok(Ok(is_anomaly)) => {
            Json(InteractionResponse {
                status: if is_anomaly { "Anomaly detected by GNN".into() } else { "Normal interaction".into() },
                is_anomaly,
            })
        }
        Ok(Err(e)) => {
            println!("Error processing interaction with GNN service: {}", e);
            Json(InteractionResponse {
                status: format!("Error: {}", e),
                is_anomaly: false,
            })
        }
        Err(e) => {
            println!("Task join error: {}", e);
            Json(InteractionResponse {
                status: format!("Task join error: {}", e),
                is_anomaly: false,
            })
        }
    }
}

//
// API Endpoint: /stats
// Returns overall network statistics (using the original method for illustration).
//

#[get("/stats")]
async fn get_stats() -> Json<StatsResponse> {
    let result = task::spawn_blocking(|| -> Result<StatsResponse, Box<dyn Error + Send + Sync>> {
        let connection_string = env::var("DATABASE_URL")?;
        let mut client = Client::connect(&connection_string, NoTls)?;
        let query = "SELECT source, target, rating, timestamp, anomaly FROM ratings";
        let rows = client.query(query, &[])?;

        // Build a simple adjacency map and compute per-node aggregated ratings.
        let mut adj_map: HashMap<i32, Vec<(f32, i64, i16)>> = HashMap::new();
        let mut node_sum: HashMap<i32, (f32, f32, u32)> = HashMap::new();

        for row in rows {
            let source: i32 = row.get("source");
            let target: i32 = row.get("target");
            let rating: f32 = row.get("rating");
            let timestamp: i64 = row.get("timestamp");
            let anomaly: i16 = row.get("anomaly");

            adj_map.entry(source).or_insert_with(Vec::new).push((rating, timestamp, anomaly));

            let update_stats = |map: &mut HashMap<i32, (f32, f32, u32)>, node: i32, rating: f32| {
                let entry = map.entry(node).or_insert((0.0, 0.0, 0));
                entry.0 += rating;
                entry.1 += rating * rating;
                entry.2 += 1;
            };

            update_stats(&mut node_sum, source, rating);
            update_stats(&mut node_sum, target, rating);
        }

        let mut total_interactions = 0;
        let mut normal_interactions = 0;
        let mut anomalous_interactions = 0;

        for (source, ratings) in &adj_map {
            if let Some(&(sum, sum_sq, count)) = node_sum.get(source) {
                let mean = sum / count as f32;
                let std_dev = if count > 1 { ((sum_sq / count as f32) - mean * mean).sqrt() } else { 0.0 };
                for (rating, _timestamp, _anomaly) in ratings {
                    total_interactions += 1;
                    let diff = (rating - mean).abs();
                    // Using a dynamic threshold when std_dev > 0, else a fixed threshold.
                    let is_anomaly = if std_dev == 0.0 {
                        rating.abs() > 2.0
                    } else {
                        diff > 2.0 * std_dev
                    };
                    if is_anomaly {
                        anomalous_interactions += 1;
                    } else {
                        normal_interactions += 1;
                    }
                }
            }
        }

        let ratio = if total_interactions > 0 {
            anomalous_interactions as f32 / total_interactions as f32
        } else {
            0.0
        };

        Ok(StatsResponse {
            total_interactions,
            normal_interactions,
            anomalous_interactions,
            anomaly_ratio: ratio,
        })
    }).await;

    match result {
        Ok(Ok(stats)) => Json(stats),
        _ => Json(StatsResponse {
            total_interactions: 0,
            normal_interactions: 0,
            anomalous_interactions: 0,
            anomaly_ratio: 0.0,
        }),
    }
}

//
// API Endpoint: /all_interactions
// Returns all interactions as JSON.
//
#[get("/all_interactions")]
async fn get_all_interactions() -> Json<Vec<InteractionRecord>> {
    let result = task::spawn_blocking(|| -> Result<Vec<InteractionRecord>, Box<dyn Error + Send + Sync>> {
        let connection_string = env::var("DATABASE_URL")?;
        let mut client = Client::connect(&connection_string, NoTls)?;
        let query = "SELECT source, target, rating, timestamp, anomaly FROM ratings";
        let rows = client.query(query, &[])?;
        let mut interactions = Vec::new();
        for row in rows {
            interactions.push(InteractionRecord {
                source: row.get("source"),
                target: row.get("target"),
                rating: row.get("rating"),
                timestamp: row.get("timestamp"),
                anomaly: row.get("anomaly"),
            });
        }
        Ok(interactions)
    }).await;

    match result {
        Ok(Ok(interactions)) => Json(interactions),
        _ => Json(vec![]),
    }
}

//
// API Endpoint: /table_page
// Serves a static HTML file (e.g., for a table view of interactions)
//
#[get("/table_page")]
async fn table_page() -> Option<rocket::fs::NamedFile> {
    rocket::fs::NamedFile::open("static/table.html").await.ok()
}

//
// Rocket Setup and CORS Configuration
//
fn build_rocket() -> Rocket<Build> {
    let allowed_origins = AllowedOrigins::all();
    let cors = CorsOptions::default()
        .allowed_origins(allowed_origins)
        .allowed_headers(rocket_cors::AllowedHeaders::some(&["Content-Type"]))
        .allow_credentials(true)
        .to_cors()
        .expect("error creating CORS fairing");

    rocket::build()
        .mount("/", routes![add_interaction, get_stats, get_all_interactions, table_page])
        .attach(cors)
}

#[launch]
fn rocket() -> _ {
    dotenv().ok(); // Load environment variables (DATABASE_URL, GNN_SERVICE_URL, etc.)
    build_rocket()
}

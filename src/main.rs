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

//
// Structures for API communication
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
// Internal structures for anomaly detection calculations
//

#[derive(Debug)]
struct Record {
    source: i32,
    target: i32,
    rating: f32,
    timestamp: i64,
    anomaly: i16, // inserted as 0 when added
}

#[derive(Debug)]
struct Edge {
    target: i32,
    rating: f32,
    timestamp: i64,
    anomaly: i16,
}

#[derive(Debug, Default)]
struct NodeStats {
    sum: f32,
    sum_sq: f32,
    count: u32,
}

impl NodeStats {
    fn update(&mut self, rating: f32) {
        self.sum += rating;
        self.sum_sq += rating * rating;
        self.count += 1;
    }
    fn mean(&self) -> f32 {
        self.sum / self.count as f32
    }
    fn std_dev(&self) -> f32 {
        let mean = self.mean();
        let variance = (self.sum_sq / self.count as f32) - mean * mean;
        variance.sqrt()
    }
}

const THRESHOLD_MULTIPLE: f32 = 2.0;
const FIXED_THRESHOLD: f32 = 2.0;

//
// Process a new interaction: insert and calculate anomaly status.
// Updated error type to Box<dyn Error + Send + Sync>
//
fn process_interaction(new_int: NewInteraction) -> Result<bool, Box<dyn Error + Send + Sync>> {
    let connection_string = env::var("DATABASE_URL")?;
    let mut client = Client::connect(&connection_string, NoTls)?;

    // Insert the new interaction using the current Unix epoch time.
    let insert_query = "INSERT INTO ratings (source, target, rating, timestamp, anomaly)
                        VALUES ($1, $2, $3, EXTRACT(EPOCH FROM NOW())::bigint, 0)";
    client.execute(insert_query, &[&new_int.source, &new_int.target, &new_int.rating])?;

    // Retrieve historical ratings for the source.
    let query = "SELECT rating FROM ratings WHERE source = $1";
    let rows = client.query(query, &[&new_int.source])?;

    let mut count = 0;
    let mut sum = 0.0_f32;
    let mut sum_sq = 0.0_f32;
    for row in rows {
        let r: f32 = row.get("rating");
        sum += r;
        sum_sq += r * r;
        count += 1;
    }
    let mean = sum / count as f32;
    let std_dev = if count > 1 {
        let variance = (sum_sq / count as f32) - (mean * mean);
        variance.sqrt()
    } else {
        0.0
    };

    // Compute anomaly based on dynamic or fixed threshold.
    let is_anomaly = if std_dev == 0.0 {
        new_int.rating.abs() > FIXED_THRESHOLD
    } else {
        (new_int.rating - mean).abs() > (THRESHOLD_MULTIPLE * std_dev)
    };

    Ok(is_anomaly)
}

//
// API endpoint to add a new interaction.
//
#[post("/add_interaction", format = "json", data = "<new_int>")]
async fn add_interaction(new_int: Json<NewInteraction>) -> Json<InteractionResponse> {
    let new_int = new_int.into_inner();
    let result = task::spawn_blocking(move || process_interaction(new_int)).await;
    match result {
        Ok(Ok(is_anomaly)) => {
            Json(InteractionResponse {
                status: if is_anomaly { "Anomaly".into() } else { "Normal".into() },
                is_anomaly,
            })
        }
        Ok(Err(e)) => {
            println!("Error processing interaction: {}", e);
            Json(InteractionResponse {
                status: format!("Error processing interaction: {}", e),
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
// API endpoint to return current network statistics.
//
#[get("/stats")]
async fn get_stats() -> Json<StatsResponse> {
    let result = task::spawn_blocking(|| -> Result<StatsResponse, Box<dyn Error + Send + Sync>> {
        let connection_string = env::var("DATABASE_URL")?;
        let mut client = Client::connect(&connection_string, NoTls)?;
        let query = "SELECT source, target, rating, timestamp, anomaly FROM ratings";
        let rows = client.query(query, &[])?;

        let mut adj_map: HashMap<i32, Vec<Edge>> = HashMap::new();
        let mut node_stats: HashMap<i32, NodeStats> = HashMap::new();

        for row in rows {
            let record = Record {
                source: row.get("source"),
                target: row.get("target"),
                rating: row.get("rating"),
                timestamp: row.get("timestamp"),
                anomaly: row.get("anomaly"),
            };
            let edge = Edge {
                target: record.target,
                rating: record.rating,
                timestamp: record.timestamp,
                anomaly: record.anomaly,
            };
            adj_map.entry(record.source)
                .or_insert_with(Vec::new)
                .push(edge);

            node_stats.entry(record.source).or_default().update(record.rating);
            node_stats.entry(record.target).or_default().update(record.rating);
        }

        let mut total_interactions = 0;
        let mut normal_interactions = 0;
        let mut anomalous_interactions = 0;
        for (source, edges) in &adj_map {
            let src_stats = match node_stats.get(source) {
                Some(stats) => stats,
                None => continue,
            };
            let src_mean = src_stats.mean();
            let src_std = src_stats.std_dev();
            for edge in edges {
                total_interactions += 1;
                let diff = (edge.rating - src_mean).abs();
                let is_anomaly = if src_std == 0.0 {
                    edge.rating.abs() > FIXED_THRESHOLD
                } else {
                    diff > (THRESHOLD_MULTIPLE * src_std)
                };
                if is_anomaly {
                    anomalous_interactions += 1;
                } else {
                    normal_interactions += 1;
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
        _ => Json(StatsResponse { total_interactions: 0, normal_interactions: 0, anomalous_interactions: 0, anomaly_ratio: 0.0 }),
    }
}

//
// API endpoint to return all interactions as JSON.
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
            let rec = InteractionRecord {
                source: row.get("source"),
                target: row.get("target"),
                rating: row.get("rating"),
                timestamp: row.get("timestamp"),
                anomaly: row.get("anomaly"),
            };
            interactions.push(rec);
        }
        Ok(interactions)
    }).await;

    match result {
        Ok(Ok(interactions)) => Json(interactions),
        _ => Json(vec![]),
    }
}

//
// API endpoint to serve a static HTML page for the complete interactions table.
//
#[get("/table_page")]
async fn table_page() -> Option<rocket::fs::NamedFile> {
    rocket::fs::NamedFile::open("static/table.html").await.ok()
}

//
// Build the Rocket application with CORS enabled.
//
fn build_rocket() -> Rocket<Build> {
    let allowed_origins = AllowedOrigins::all();
    let cors = CorsOptions {
        allowed_origins,
        allowed_headers: rocket_cors::AllowedHeaders::some(&["Content-Type"]),
        allow_credentials: true,
        ..Default::default()
    }
        .to_cors()
        .expect("error creating CORS fairing");

    rocket::build()
        .mount("/", routes![add_interaction, get_stats, get_all_interactions, table_page])
        .attach(cors)
}

#[launch]
fn rocket() -> _ {
    dotenv().ok(); // load .env early
    build_rocket()
}

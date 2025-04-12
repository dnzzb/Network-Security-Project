use postgres::{Client, NoTls};
use std::collections::HashMap;
use std::error::Error;
use std::env;
use dotenv::dotenv;

/// Structure for holding a record from the database.
#[derive(Debug)]
struct Record {
    source: i32,      // PostgreSQL int4 maps to i32.
    target: i32,      // PostgreSQL int4 maps to i32.
    rating: f32,
    timestamp: i64,   // PostgreSQL BIGINT maps to i64.
    anomaly: i16,     // PostgreSQL SMALLINT maps to i16.
}

/// Represents an edge in the graph.
#[derive(Debug)]
struct Edge {
    target: i32,
    rating: f32,
    timestamp: i64,
    anomaly: i16,
}

/// NodeStats accumulates sum, sum of squares, and count per node.
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

/// The dynamic threshold multiplier for anomaly detection.
const THRESHOLD_MULTIPLE: f32 = 2.0;

/// A fixed threshold used for nodes with only one interaction (zero standard deviation).
const FIXED_THRESHOLD: f32 = 2.0;

fn main() -> Result<(), Box<dyn Error>> {
    // Load environment variables from the `.env` file.
    dotenv().ok();

    // Retrieve the connection string from the environment.
    let connection_string = env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set in your .env file or environment");

    // Connect to the Supabase PostgreSQL database.
    let mut client = Client::connect(&connection_string, NoTls)?;
    println!("Connected to Supabase PostgreSQL database successfully!");

    // Query to select all records from the ratings table.
    let query = "SELECT source, target, rating, timestamp, anomaly FROM ratings";
    let rows = client.query(query, &[])?;

    // Build an adjacency map and per-node statistics.
    let mut adj_map: HashMap<i32, Vec<Edge>> = HashMap::new();
    let mut node_stats: HashMap<i32, NodeStats> = HashMap::new();

    // Counters for overall classification.
    let mut total_interactions = 0;
    let mut normal_interactions = 0;
    let mut anomalous_interactions = 0;

    // Process each row.
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

        // Insert edge into the adjacency map keyed by source.
        adj_map.entry(record.source)
            .or_insert_with(Vec::new)
            .push(edge);

        // Update node stats for both source and target nodes.
        node_stats.entry(record.source).or_default().update(record.rating);
        node_stats.entry(record.target).or_default().update(record.rating);

        total_interactions += 1;
    }

    // Print per-node trust statistics.
    println!("Node Trust Ratios and Standard Deviations:");
    for (node, stats) in &node_stats {
        println!(
            "  Node {}: Mean = {:.2}, StdDev = {:.2} ({} interactions)",
            node,
            stats.mean(),
            stats.std_dev(),
            stats.count
        );
    }

    // Analyze each edge for anomaly detection.
    println!("\nEdge Anomaly Classification using Dynamic Thresholding:");
    for (source, edges) in &adj_map {
        let src_stats = node_stats.get(source).unwrap();
        let src_mean = src_stats.mean();
        let src_std = src_stats.std_dev();

        for edge in edges {
            let tgt_stats = node_stats.get(&edge.target).unwrap();
            let tgt_mean = tgt_stats.mean();
            let tgt_std = tgt_stats.std_dev();

            // Calculate absolute differences.
            let src_diff = (edge.rating - src_mean).abs();
            let tgt_diff = (edge.rating - tgt_mean).abs();

            // If the standard deviation is 0, use the fixed threshold; else use dynamic thresholding.
            let anomalous_src = if src_std == 0.0 {
                edge.rating.abs() > FIXED_THRESHOLD
            } else {
                src_diff > (THRESHOLD_MULTIPLE * src_std)
            };

            let anomalous_tgt = if tgt_std == 0.0 {
                edge.rating.abs() > FIXED_THRESHOLD
            } else {
                tgt_diff > (THRESHOLD_MULTIPLE * tgt_std)
            };

            // Mark as anomaly if either node flags the interaction.
            let status = if anomalous_src || anomalous_tgt { "Anomaly" } else { "Normal" };

            // Update counters.
            if status == "Anomaly" {
                anomalous_interactions += 1;
            } else {
                normal_interactions += 1;
            }

            println!(
                "Source: {:4} -> Target: {:4} | Rating: {:5.2} | Src Mean = {:5.2}, Diff = {:5.2} | Tgt Mean = {:5.2}, Diff = {:5.2} => {}",
                source, edge.target, edge.rating, src_mean, src_diff, tgt_mean, tgt_diff, status
            );
        }
    }

    // Print summary statistics.
    println!("\nSummary:");
    println!("Total interactions: {}", total_interactions);
    println!("Normal interactions: {}", normal_interactions);
    println!("Anomalous interactions: {}", anomalous_interactions);
    if total_interactions > 0 {
        let ratio = anomalous_interactions as f32 / total_interactions as f32;
        println!("Ratio (anomalous/total): {:.2}", ratio);
    }

    Ok(())
}

use anyhow::Result;
use std::time::Instant;
use std::fs;
use std::env;

use tonic::transport::{Certificate, Channel, ClientTlsConfig};
use tonic::{Request, Status};

use pytja_proto::pytja::pytja_service_client::PytjaServiceClient;
use pytja_proto::pytja::{
    CreateVectorCollectionRequest,
    DeleteVectorCollectionRequest,
    ListVectorCollectionsRequest,
    VectorUpsertRequest,
    VectorSearchRequest,
    VectorGetRequest,
    VectorDeleteRequest,
    VectorCountRequest,
    VectorPointData,
};

fn auth_interceptor(mut req: Request<()>) -> Result<Request<()>, Status> {
    let token = env::var("PYTJA_TOKEN").unwrap_or_else(|_| "PLATZHALTER_TOKEN".to_string());
    let auth_header = format!("Bearer {}", token);
    match auth_header.parse() {
        Ok(meta) => {
            req.metadata_mut().insert("authorization", meta);
            Ok(req)
        }
        Err(_) => Err(Status::unauthenticated("Invalid token format")),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("==============================================");
    println!("  PYTJA SqliteVec E2E TEST");
    println!("  Embedded Vector Store — Pure Rust Engine");
    println!("==============================================\n");

    // --- CONNECT ---
    let pem = fs::read_to_string("certs/server.crt").expect("Certificate not found!");
    let ca = Certificate::from_pem(pem);
    let tls = ClientTlsConfig::new()
        .ca_certificate(ca)
        .domain_name("localhost");

    let channel = Channel::from_static("https://127.0.0.1:50051")
        .tls_config(tls)?
        .connect()
        .await?;

    let mut client = PytjaServiceClient::with_interceptor(channel, auth_interceptor);
    println!("[OK] Connected via gRPC+TLS\n");

    let mount = "local_vectors".to_string();
    let collection = "sqlite_vec_test".to_string();

    // =================================================================
    // TEST 1: Create Collection
    // =================================================================
    println!("--- TEST 1: Create Collection ---");
    let start = Instant::now();
    let resp = client.create_vector_collection(CreateVectorCollectionRequest {
        mount_name: mount.clone(),
        collection_name: collection.clone(),
        dimension: 4,
        distance_metric: "cosine".to_string(),
    }).await?;
    println!("[OK] {} ({}ms)\n", resp.into_inner().message, start.elapsed().as_millis());

    // =================================================================
    // TEST 2: Upsert Vectors
    // =================================================================
    println!("--- TEST 2: Upsert 5 Vectors ---");
    let start = Instant::now();
    let resp = client.vector_upsert(VectorUpsertRequest {
        mount_name: mount.clone(),
        collection_name: collection.clone(),
        points: vec![
            VectorPointData {
                id: "vec_ai".to_string(),
                embedding: vec![0.9, 0.1, 0.05, 0.02],
                metadata_json: r#"{"topic": "artificial_intelligence", "lang": "en"}"#.to_string(),
            },
            VectorPointData {
                id: "vec_db".to_string(),
                embedding: vec![0.1, 0.9, 0.05, 0.01],
                metadata_json: r#"{"topic": "databases", "lang": "en"}"#.to_string(),
            },
            VectorPointData {
                id: "vec_rust".to_string(),
                embedding: vec![0.3, 0.3, 0.8, 0.1],
                metadata_json: r#"{"topic": "rust_programming", "lang": "en"}"#.to_string(),
            },
            VectorPointData {
                id: "vec_ki_de".to_string(),
                embedding: vec![0.85, 0.15, 0.1, 0.05],
                metadata_json: r#"{"topic": "kuenstliche_intelligenz", "lang": "de"}"#.to_string(),
            },
            VectorPointData {
                id: "vec_crypto".to_string(),
                embedding: vec![0.05, 0.1, 0.2, 0.95],
                metadata_json: r#"{"topic": "cryptography", "lang": "en"}"#.to_string(),
            },
        ],
    }).await?;
    let inner = resp.into_inner();
    println!("[OK] {} ({}ms)\n", inner.message, start.elapsed().as_millis());

    // =================================================================
    // TEST 3: Vector Count
    // =================================================================
    println!("--- TEST 3: Vector Count ---");
    let resp = client.vector_count(VectorCountRequest {
        mount_name: mount.clone(),
        collection_name: collection.clone(),
    }).await?;
    let count = resp.into_inner().count;
    println!("[OK] Collection has {} vectors\n", count);
    assert_eq!(count, 5, "Expected 5 vectors!");

    // =================================================================
    // TEST 4: Similarity Search (Cosine)
    // =================================================================
    println!("--- TEST 4: Similarity Search (query similar to AI) ---");
    let start = Instant::now();
    let resp = client.vector_search(VectorSearchRequest {
        mount_name: mount.clone(),
        collection_name: collection.clone(),
        query_vector: vec![0.88, 0.12, 0.08, 0.03],  // Similar to AI vectors
        top_k: 3,
        filter_json: String::new(),
    }).await?;
    let search = resp.into_inner();
    println!("[OK] {} hits in {:.2}ms", search.total_hits, search.search_time_ms);
    for (i, hit) in search.hits.iter().enumerate() {
        println!("  #{}: id={}, score={:.4}, meta={}", i + 1, hit.id, hit.score, hit.metadata_json);
    }
    // The top result should be vec_ai or vec_ki_de (both are AI-related)
    assert!(
        search.hits[0].id == "vec_ai" || search.hits[0].id == "vec_ki_de",
        "Expected AI-related vector as top result, got: {}",
        search.hits[0].id
    );
    println!();

    // =================================================================
    // TEST 5: Search with Metadata Filter
    // =================================================================
    println!("--- TEST 5: Search with Metadata Filter (lang=de only) ---");
    let start = Instant::now();
    let resp = client.vector_search(VectorSearchRequest {
        mount_name: mount.clone(),
        collection_name: collection.clone(),
        query_vector: vec![0.88, 0.12, 0.08, 0.03],
        top_k: 5,
        filter_json: r#"{"lang": "de"}"#.to_string(),
    }).await?;
    let search = resp.into_inner();
    println!("[OK] {} hits in {:.2}ms", search.total_hits, search.search_time_ms);
    for (i, hit) in search.hits.iter().enumerate() {
        println!("  #{}: id={}, score={:.4}, meta={}", i + 1, hit.id, hit.score, hit.metadata_json);
    }
    assert_eq!(search.total_hits, 1, "Expected exactly 1 German result!");
    assert_eq!(search.hits[0].id, "vec_ki_de");
    println!();

    // =================================================================
    // TEST 6: Get by ID
    // =================================================================
    println!("--- TEST 6: Get Vectors by ID ---");
    let resp = client.vector_get(VectorGetRequest {
        mount_name: mount.clone(),
        collection_name: collection.clone(),
        ids: vec!["vec_rust".to_string(), "vec_crypto".to_string()],
    }).await?;
    let points = resp.into_inner().points;
    println!("[OK] Retrieved {} points", points.len());
    for p in &points {
        println!("  id={}, dim={}, meta={}", p.id, p.embedding.len(), p.metadata_json);
    }
    assert_eq!(points.len(), 2);
    println!();

    // =================================================================
    // TEST 7: Delete by ID
    // =================================================================
    println!("--- TEST 7: Delete vec_crypto ---");
    let resp = client.vector_delete(VectorDeleteRequest {
        mount_name: mount.clone(),
        collection_name: collection.clone(),
        ids: vec!["vec_crypto".to_string()],
        filter_json: String::new(),
    }).await?;
    println!("[OK] {}\n", resp.into_inner().message);

    // Verify count is now 4
    let resp = client.vector_count(VectorCountRequest {
        mount_name: mount.clone(),
        collection_name: collection.clone(),
    }).await?;
    let count = resp.into_inner().count;
    println!("[OK] Count after delete: {} (expected 4)\n", count);
    assert_eq!(count, 4);

    // =================================================================
    // TEST 8: List Collections
    // =================================================================
    println!("--- TEST 8: List Collections ---");
    let resp = client.list_vector_collections(ListVectorCollectionsRequest {
        mount_name: mount.clone(),
    }).await?;
    let cols = resp.into_inner().collections;
    println!("[OK] {} collection(s):", cols.len());
    for c in &cols {
        println!("  name={}, dim={}, metric={}, vectors={}", c.name, c.dimension, c.distance_metric, c.vector_count);
    }
    println!();

    // =================================================================
    // TEST 9: Cleanup — Delete Collection
    // =================================================================
    println!("--- TEST 9: Cleanup ---");
    let resp = client.delete_vector_collection(DeleteVectorCollectionRequest {
        mount_name: mount.clone(),
        collection_name: collection.clone(),
    }).await?;
    println!("[OK] {}\n", resp.into_inner().message);

    // =================================================================
    // RESULT
    // =================================================================
    println!("==============================================");
    println!("  ALL 9 TESTS PASSED");
    println!("  SqliteVec Embedded Vector Store: WORKING");
    println!("==============================================");

    Ok(())
}

use anyhow::Result;
use std::time::Instant;
use std::fs;
use std::env;

use tonic::transport::{Certificate, Channel, ClientTlsConfig};
use tonic::{Request, Status};

use pytja_proto::pytja::pytja_service_client::PytjaServiceClient;
use pytja_proto::pytja::{
    CreateVectorCollectionRequest,
    VectorUpsertRequest,
    VectorSearchRequest,
    VectorPointData
};

fn auth_interceptor(mut req: Request<()>) -> Result<Request<()>, Status> {
    let token = env::var("PYTJA_TOKEN").unwrap_or_else(|_| "PLATZHALTER_TOKEN".to_string());
    let auth_header = format!("Bearer {}", token);

    match auth_header.parse() {
        Ok(meta) => {
            req.metadata_mut().insert("authorization", meta);
            Ok(req)
        }
        Err(_) => Err(Status::unauthenticated("Ungueltiges Token-Format")),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("--- PYTJA VECTOR STORE E2E TEST ---");

    let pem = fs::read_to_string("certs/server.crt").expect("Zertifikat nicht gefunden!");
    let ca = Certificate::from_pem(pem);

    let tls = ClientTlsConfig::new()
        .ca_certificate(ca)
        .domain_name("localhost");

    let channel = Channel::from_static("https://127.0.0.1:50051")
        .tls_config(tls)?
        .connect()
        .await?;

    let mut client = PytjaServiceClient::with_interceptor(channel, auth_interceptor);
    println!("VERBUNDEN: gRPC Channel etabliert (TLS & Auth Interceptor aktiv).");

    let mount_name = "vectors".to_string();
    let collection_name = "ai_memory".to_string();

    println!("Erstelle Collection '{}'...", collection_name);
    let start = Instant::now();
    client.create_vector_collection(CreateVectorCollectionRequest {
        mount_name: mount_name.clone(),
        collection_name: collection_name.clone(),
        dimension: 3,
        distance_metric: "cosine".to_string(),
    }).await?;
    println!("COLLECTION BEREIT ({}ms)", start.elapsed().as_millis());

    println!("Injiziere Test-Vektoren...");
    let start = Instant::now();
    client.vector_upsert(VectorUpsertRequest {
        mount_name: mount_name.clone(),
        collection_name: collection_name.clone(),
        points: vec![
            VectorPointData {
                id: "doc_1".to_string(),
                embedding: vec![0.1, 0.2, 0.3],
                metadata_json: "{\"text\": \"Kuenstliche Intelligenz\"}".to_string(),
            },
            VectorPointData {
                id: "doc_2".to_string(),
                embedding: vec![0.9, 0.8, 0.7],
                metadata_json: "{\"text\": \"Datenbank Architektur\"}".to_string(),
            },
        ],
    }).await?;
    println!("UPSERT ERFOLGREICH ({}ms)", start.elapsed().as_millis());

    println!("Fuehre semantische Suche aus...");
    let start = Instant::now();
    let response = client.vector_search(VectorSearchRequest {
        mount_name: mount_name.clone(),
        collection_name: collection_name.clone(),
        query_vector: vec![0.15, 0.25, 0.35],
        top_k: 1,
        filter_json: String::new(),
    }).await?;

    let search_time = start.elapsed().as_millis();

    let hits = response.into_inner().hits;
    println!("SUCHE ERFOLGREICH ({}ms)", search_time);

    for (i, res) in hits.iter().enumerate() {
        println!("  Treffer {}: ID = {}, Score = {:.4}", i + 1, res.id, res.score);
        println!("  Payload: {}", res.metadata_json);
    }

    Ok(())
}
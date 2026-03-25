pub mod pytja {
    tonic::include_proto!("pytja");
}

pub use pytja::pytja_service_server::{PytjaService, PytjaServiceServer};
pub use pytja::pytja_service_client::PytjaServiceClient;

pub use pytja::{
    PingRequest, PingResponse,
    ListRequest, ListResponse,
    FileInfo,
    CreateNodeRequest, ActionResponse,
    ReadFileRequest, ReadFileResponse,
    DeleteNodeRequest, MoveNodeRequest,
    CopyNodeRequest, ChangeModeRequest,
    ChownRequest, LockRequest,
    UsageRequest, UsageResponse,
    FindRequest, FindResponse,
    GrepRequest, GrepResponse,
    StatRequest, StatResponse,
    TreeRequest, TreeResponse,
    UploadRequest, FileMetadata,
    DownloadRequest, FileChunk,
    ExecRequest, ExecResponse,
    ChallengeRequest, ChallengeResponse,
    LoginRequest, LoginResponse
};
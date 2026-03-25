use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum RadarPermission {
    #[serde(rename = "fs_read")]
    FsRead,
    #[serde(rename = "fs_write")]
    FsWrite,
    #[serde(rename = "network_tcp")]
    NetworkTcp,
    #[serde(rename = "radar_ipc")]
    RadarIpc,
    #[serde(rename = "admin")]
    Admin,
    #[serde(rename = "display_ui")]
    DisplayUi, // Das neue UI-Recht
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    pub name: String,
    pub version: String,
    pub description: String,
    pub permissions: Vec<RadarPermission>,
    #[serde(default)]
    pub autostart: bool,
}
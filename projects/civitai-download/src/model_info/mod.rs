use std::path::Path;

pub use self::model_one::ModelInfo;

mod model_one;

#[derive(Clone, Debug)]
pub struct RequestModel {
    pub id: usize,
}

impl RequestModel {
    pub fn new(id: usize) -> Self {
        Self { id }
    }
    /// Get the URL for this request
    ///
    /// <https://github.com/civitai/civitai/wiki/REST-API-Reference#get-apiv1modelsmodelid>
    pub async fn send(&self) -> Result<ModelInfo, reqwest::Error> {
        reqwest::get(self.url()).await?.json().await
    }
    pub fn url(&self) -> String {
        format!("https://civitai.com/api/v1/models/{}", self.id)
    }
}

impl ModelInfo {
    /// Get the primary download link for this model
    ///
    /// Primary model is the latest model in the list
    pub fn download_link(&self) -> &str {
        match self.model_versions.first() {
            Some(s) => s.download_url.as_str(),
            None => "",
        }
    }
    /// Download the primary model
    ///
    /// Primary model is the latest model in the list
    pub fn download(&self, local: &Path) -> Result<(), reqwest::Error> {
        todo!()
        // let mut resp = reqwest::get(self.download_link())?;
        // let mut out = std::fs::File::create(format!("{}.zip", self.name))?;
        // std::io::copy(&mut resp, &mut out)?;
        // Ok(())
    }
}

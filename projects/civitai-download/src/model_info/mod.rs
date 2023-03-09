pub use self::model_one::ModelInfo;

mod model_one;

#[derive(Clone, Debug)]
pub struct RequestModel {
    pub id: usize,
}

impl RequestModel {
    pub fn new(id: usize) -> Self {
        Self {
            id,
        }
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
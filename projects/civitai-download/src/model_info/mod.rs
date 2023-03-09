use trauma::download::Download;

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
    /// It creates a new download task and returns a handle to it.
    ///
    /// Primary model is the latest model in the list
    pub fn download(&self, local: &str) -> Result<Download, trauma::Error> {
        let mut file_name = local;
        let download_link = match self.model_versions.first() {
            Some(s) => {
                if file_name.is_empty() {
                    file_name = s.name.as_str()
                }
                s.download_url.as_str()
            }
            None => Err(trauma::Error::InvalidUrl("Missing download link".to_string()))?,
        };
        let mut task = Download::try_from(download_link)?;
        task.filename = file_name.to_string();
        Ok(task)
    }
}

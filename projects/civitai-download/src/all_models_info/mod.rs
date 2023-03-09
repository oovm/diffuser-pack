use reqwest::Url;

pub use self::defines::AllModels;

mod defines;

#[derive(Clone, Debug)]
pub struct RequestAllModels {
    pub page: usize,
}

impl Default for RequestAllModels {
    fn default() -> Self {
        Self { page: 1 }
    }
}

impl RequestAllModels {
    pub fn new(page: usize) -> Self {
        Self { page, ..Default::default() }
    }

    pub fn with_page(mut self, page: usize) -> Self {
        self.page = page;
        self
    }
    /// Get the next page of all_models_info
    ///
    /// <https://github.com/civitai/civitai/wiki/REST-API-Reference#get-apiv1models>
    pub async fn send(&self) -> Result<AllModels, reqwest::Error> {
        reqwest::get(self.url()).await?.json().await
    }
    pub fn url(&self) -> String {
        format!("https://civitai.com/api/v1/models?page={page}", page = self.page)
    }
}

impl AllModels {
    /// Get the next page of all_models_info
    pub fn next_page(&self) -> Result<RequestAllModels, reqwest::Error> {
        let mut out = RequestAllModels::default();
        match Url::parse(&self.metadata.next_page) {
            Ok(o) => {
                for (key, value) in o.query_pairs() {
                    match key.as_ref() {
                        "page" => match value.parse() {
                            Ok(o) => out.page = o,
                            Err(_) => {}
                        },
                        _ => {
                            println!("Unknown key: {} => {}", key, value);
                        }
                    }
                }
            }
            Err(_) => {
                todo!()
            }
        }
        Ok(out)
    }
}

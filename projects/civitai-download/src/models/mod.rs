pub use self::model_all::AllModels;

mod model_all;
mod model_one;

// https://civitai.com/api/v1/models
pub async fn all_models(page: usize) -> Result<AllModels, reqwest::Error> {
    reqwest::get("https://civitai.com/api/v1/models").await?
        .json().await
}


pub async fn find_model(id: usize) {}
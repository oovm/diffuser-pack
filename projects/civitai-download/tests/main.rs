use civitai::{RequestAllModels, RequestModel};

#[test]
fn ready() {
    println!("it works!")
}

#[tokio::test]
async fn test_all() {
    let models = RequestAllModels::default().send().await.unwrap();
    println!("{:#?}", models.next_page());
}

#[tokio::test]
async fn test_model1() {
    let models = RequestModel::new(1).send().await.unwrap();
    println!("{:#?}", models);
}
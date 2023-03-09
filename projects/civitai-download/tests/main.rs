use civitai::all_models;

#[test]
fn ready() {
    println!("it works!")
}

#[tokio::test]
async fn test_all() {
    let models = all_models().await.unwrap();
    println!("{:#?}", models);
}
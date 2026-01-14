

def debug_show_pca_of_embedding(embeddings):
    from sklearn.decomposition import PCA
    import plotly.express as px
    pca = PCA(n_components=3, whiten=True)
    h, w, n_features = embeddings.shape
    output_cpu = embeddings.flatten(end_dim=1).cpu().numpy()
    projection = pca.fit_transform(output_cpu).reshape(h, w, 3)
    projection = (projection - projection.min()) / (projection.max() - projection.min())
    px.imshow(projection).show()


def debug_show_image(image):
    import plotly.express as px
    px.imshow(image).show()

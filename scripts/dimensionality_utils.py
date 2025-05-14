from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def reduce_dim_pca(X, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X.toarray())

def reduce_dim_tsne(X, n_components=2, perplexity=30):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    return tsne.fit_transform(X.toarray())

import anndata as an
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import squidpy as sq
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsTransformer


def filter_and_run_scanpy(adata: an.AnnData, min_counts: int = 10, resolution: float = 0.5) -> an.AnnData:
    """Filter and run scanpy analysis.

    Parameters
    ----------
    adata
        Anndata object.
    min_counts
        minimum transcript count to keep cell.

    Returns
    -------
    Anndata analyzed object.
    """
    sc.pp.filter_cells(adata, min_counts=min_counts, inplace=True)

    print("total cells=", adata.shape[0])
    print("mean transcripts per cell=", adata.obs["barcodeCount"].mean())
    print("median transcripts per cell=", adata.obs["barcodeCount"].median())

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=10)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution, key_added="clusters")

    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    sc.pl.embedding(adata, "umap", color="clusters", ax=axs[0], show=False)
    sq.pl.spatial_scatter(adata, color="clusters", shape=None, size=1, ax=axs[1])
    plt.tight_layout()

    return adata


def annotate(
    ad_spatial: an.AnnData,
    ad_ref: an.AnnData,
    label_ref: str = "celltype",
    label_key: str = "celltype",
    metaref2add: str = None,
) -> an.AnnData:
    """Transfert cell type label from single cell to spatial annData object using SCVI (scArches developpers credits for final knn neighbor transfert)

    Parameters
    ----------
    ad_spatial
        Anndata spatial object.
    ad_ref
        Anndata single-cell reference object.
    label_ref
        .obs key in single-cell reference object.
    label_key
        .obs key in spatial object.
    metaref2add
        .obs key in single-cell reference object to transfert to spatial.

    Returns
    -------
    Anndata labeled object.
    """
    ad_spatial.var.index = ad_spatial.var.index.str.upper()  # lower
    ad_ref.var.index = ad_ref.var.index.str.upper()

    print("ref. ", ad_ref.shape)
    print("viz. ", ad_spatial.shape)

    # Select shared gene panel genes only
    genes_Vizgen = ad_spatial.var.index
    genes_10x = ad_ref.var.index
    genes_shared = genes_Vizgen.intersection(genes_10x)  # List of shared genes
    ad_emb = ad_spatial[:, genes_Vizgen.isin(genes_shared)].copy()
    ad_ref = ad_ref[:, genes_10x.isin(genes_shared)]

    print(len(genes_shared), "common genes")

    # missed = list(set(genes_Vizgen) - set(genes_shared))
    # print("gene missed = ", missed)

    # Concatenate the datasets
    concat = ad_ref.concatenate(ad_emb, batch_key="tech", batch_categories=["10x", "MERFISH"]).copy()
    # concat.layers["counts"] = concat.X.copy()
    # sc.pp.normalize_total(concat, target_sum=1e4)
    # sc.pp.log1p(concat)
    # concat.raw = concat  # keep full dimension safe

    # Use the annotations from the 10x, and treat the MERFISH as unlabeled
    concat.obs[f"{label_key}"] = "nan"
    mask = concat.obs["tech"] == "10x"
    concat.obs[f"{label_key}"][mask] = ad_ref.obs[label_ref].values

    # Create the scVI latent space
    scvi.model.SCVI.setup_anndata(concat, layer="counts", batch_key="tech")
    vae = scvi.model.SCVI(concat)
    # Train the model
    vae.train()

    # Register the object and run scANVI
    scvi.model.SCANVI.setup_anndata(
        concat,
        layer="counts",
        batch_key="tech",
        labels_key=label_key,
        unlabeled_category="nan",
    )

    lvae = scvi.model.SCANVI.from_scvi_model(vae, labels_key=label_key, unlabeled_category="nan", adata=concat)
    lvae.train(max_epochs=20, n_samples_per_label=100)

    concat.obsm["X_scANVI"] = lvae.get_latent_representation(concat)
    adtmp = an.AnnData(X=concat.obsm["X_scANVI"], obs=concat.obs)
    adtmp = transfer_label(adtmp, label_key, "nan")

    # ref = adtmp[adtmp.obs['tech'] == "10x",:]
    # print(np.all(ref.obs[label_key] == ref.obs[label_key+'_pred']))

    plt.rcParams["figure.figsize"] = [10, 3]
    # Cell type frequencies
    # Now plot the fraction of cell types for each technology
    celltypes = adtmp.obs.loc[:, ["tech", f"{label_key}_pred"]].copy()
    celltype_counts = celltypes.groupby(by=["tech", f"{label_key}_pred"]).size().unstack().transpose()
    celltype_frac = celltype_counts / celltype_counts.sum(axis=0)
    celltype_frac.plot.bar(title="proportion")
    plt.xticks(rotation=45, ha="right")
    plt.show()

    # celltype_frac.plot.scatter(x='10x',y='MERFISH')

    merfish_mask = concat.obs["tech"] == "MERFISH"
    ad_spatial.obs[f"{label_key}"] = adtmp.obs[f"{label_key}_pred"][merfish_mask].values
    ad_spatial.obs[f"{label_key}_uncertainty"] = adtmp.obs[f"{label_key}_uncertainty"][merfish_mask].values

    if metaref2add:
        d = pd.Series(ad_ref.obs[f"{metaref2add}"].values, index=ad_ref.obs[f"{label_ref}"]).to_dict()
        ad_spatial.obs[f"{metaref2add}"] = ad_spatial.obs[f"{label_key}"].map(d)

    return ad_spatial


def transfer_label(adata: an.AnnData, label_key: str, unlabeled_category: str = "nan") -> an.AnnData:
    """ScArches developpers method for transfer label

    Parameters
    ----------
    adata
        Anndata object.
    label_key
        label to annotate.
    unlabeled_category
        category to label

    Returns
    -------
    Anndata labeled object.
    """
    reference_embedding = adata[adata.obs[label_key] != unlabeled_category, :]
    query_embedding = adata[adata.obs[label_key] == unlabeled_category, :]

    # run k-neighbors transformer
    k_neighbors_transformer = weighted_knn_trainer(
        train_adata=reference_embedding,
        train_adata_emb="X",  # location of our joint embedding
        n_neighbors=50,
    )
    # perform label transfer
    labels, uncert = weighted_knn_transfer(
        query_adata=query_embedding,
        query_adata_emb="X",  # location of our joint embedding
        label_keys=label_key,
        knn_model=k_neighbors_transformer,
        ref_adata_obs=reference_embedding.obs,
    )

    adata.obs[f"{label_key}_pred"] = adata.obs[label_key]
    adata.obs.loc[labels.index, f"{label_key}_pred"] = labels[label_key]
    adata.obs[f"{label_key}_uncertainty"] = uncert[label_key].astype("float64")

    return adata


def weighted_knn_trainer(train_adata, train_adata_emb, n_neighbors=50):
    """Trains a weighted KNN classifier on ``train_adata``.

    Parameters
    ----------
    train_adata: :class:`~anndata.AnnData`
        Annotated dataset to be used to train KNN classifier with ``label_key`` as the target variable.
    train_adata_emb: str
        Name of the obsm layer to be used for calculation of neighbors. If set to "X", anndata.X will be
        used
    n_neighbors: int
        Number of nearest neighbors in KNN classifier.
    """
    print(
        f"Weighted KNN with n_neighbors = {n_neighbors} ... ",
        end="",
    )
    k_neighbors_transformer = KNeighborsTransformer(
        n_neighbors=n_neighbors,
        mode="distance",
        algorithm="brute",
        metric="euclidean",
        n_jobs=-1,
    )
    if train_adata_emb == "X":
        train_emb = train_adata.X
    elif train_adata_emb in train_adata.obsm.keys():
        train_emb = train_adata.obsm[train_adata_emb]
    else:
        raise ValueError("train_adata_emb should be set to either 'X' or the name of the obsm layer to be used!")
    k_neighbors_transformer.fit(train_emb)
    return k_neighbors_transformer


def weighted_knn_transfer(
    query_adata,
    query_adata_emb,
    ref_adata_obs,
    label_keys,
    knn_model,
    threshold=1,
    pred_unknown=False,
    mode="package",
):
    """Annotates ``query_adata`` cells with an input trained weighted KNN classifier.

    Parameters
    ----------
    query_adata: :class:`~anndata.AnnData`
        Annotated dataset to be used to queryate KNN classifier. Embedding to be used
    query_adata_emb: str
        Name of the obsm layer to be used for label transfer. If set to "X",
        query_adata.X will be used
    ref_adata_obs: :class:`pd.DataFrame`
        obs of ref Anndata
    label_keys: str
        Names of the columns to be used as target variables (e.g. cell_type) in ``query_adata``.
    knn_model: :class:`~sklearn.neighbors._graph.KNeighborsTransformer`
        knn model trained on reference adata with weighted_knn_trainer function
    threshold: float
        Threshold of uncertainty used to annotating cells as "Unknown". cells with
        uncertainties higher than this value will be annotated as "Unknown".
        Set to 1 to keep all predictions. This enables one to later on play
        with thresholds.
    pred_unknown: bool
        ``False`` by default. Whether to annotate any cell as "unknown" or not.
        If `False`, ``threshold`` will not be used and each cell will be annotated
        with the label which is the most common in its ``n_neighbors`` nearest cells.
    mode: str
        Has to be one of "paper" or "package". If mode is set to "package",
        uncertainties will be 1 - P(pred_label), otherwise it will be 1 - P(true_label).
    """
    if not type(knn_model) == KNeighborsTransformer:
        raise ValueError("knn_model should be of type sklearn.neighbors._graph.KNeighborsTransformer!")

    if query_adata_emb == "X":
        query_emb = query_adata.X
    elif query_adata_emb in query_adata.obsm.keys():
        query_emb = query_adata.obsm[query_adata_emb]
    else:
        raise ValueError("query_adata_emb should be set to either 'X' or the name of the obsm layer to be used!")
    top_k_distances, top_k_indices = knn_model.kneighbors(X=query_emb)

    stds = np.std(top_k_distances, axis=1)
    stds = (2.0 / stds) ** 2
    stds = stds.reshape(-1, 1)

    top_k_distances_tilda = np.exp(-np.true_divide(top_k_distances, stds))

    weights = top_k_distances_tilda / np.sum(top_k_distances_tilda, axis=1, keepdims=True)
    cols = ref_adata_obs.columns[ref_adata_obs.columns.str.startswith(label_keys)]
    uncertainties = pd.DataFrame(columns=cols, index=query_adata.obs_names)
    pred_labels = pd.DataFrame(columns=cols, index=query_adata.obs_names)
    for i in range(len(weights)):
        for j in cols:
            y_train_labels = ref_adata_obs[j].values
            unique_labels = np.unique(y_train_labels[top_k_indices[i]])
            best_label, best_prob = None, 0.0
            for candidate_label in unique_labels:
                candidate_prob = weights[i, y_train_labels[top_k_indices[i]] == candidate_label].sum()
                if best_prob < candidate_prob:
                    best_prob = candidate_prob
                    best_label = candidate_label

            if pred_unknown:
                if best_prob >= threshold:
                    pred_label = best_label
                else:
                    pred_label = "Unknown"
            else:
                pred_label = best_label

            if mode == "package":
                uncertainties.iloc[i][j] = max(1 - best_prob, 0)

            else:
                raise Exception("Inquery Mode!")

            pred_labels.iloc[i][j] = pred_label

    print("finished!")

    return pred_labels, uncertainties

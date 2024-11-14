# %%
cent_distances = [500, 1000, 2000, 5000]
lu_distances = [100, 200, 500]
hwy_target_tags = [
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "residential",
]
dfs = {}
clean_dists = [(15, 8)]
# clean_dists = [0, 2, 4, 6, 8, 10, 12, 16, 20, 24]

# %%
for dist_par, dist_cons in clean_dists:
    # load a fresh copy of prems
    # assignments will change across iterations
    prems = gpd.read_file("temp/premises_clean.gpkg")
    prems.index = prems.index.astype(str)
    # simplification by current distance
    nx_simpl = graphs.nx_split_opposing_geoms(
        nx_basic_clean,
        buffer_dist=dist_par,
        prioritise_by_hwy_tag=True,
        osm_hwy_target_tags=hwy_target_tags,
        osm_matched_tags_only=True,
    )
    nx_simpl = graphs.nx_consolidate_nodes(
        nx_simpl,
        buffer_dist=dist_cons,
        crawl=True,
        prioritise_by_hwy_tag=True,
        osm_hwy_target_tags=hwy_target_tags,
        osm_matched_tags_only=True,
    )
    # to dual
    G_nx_dual = graphs.nx_to_dual(nx_simpl)
    # set live nodes for nodes within boundary
    for nd_key, nd_data in tqdm(G_nx_dual.nodes(data=True), total=G_nx_dual.number_of_nodes()):
        point = geometry.Point(nd_data["x"], nd_data["y"])
        G_nx_dual.nodes[nd_key]["live"] = point.intersects(bounds)
    # network structure
    nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(
        G_nx_dual, crs=WORKING_CRS
    )
    # # create separate network structure with length weighted data
    # for nd_key, nd_data in tqdm(
    #     G_nx_dual.nodes(data=True), total=G_nx_dual.number_of_nodes()
    # ):
    #     G_nx_dual.nodes[nd_key]["weight"] = nd_data["primal_edge"].length
    # # extract length weighted structure
    # nodes_gdf, edges_gdf, network_structure_len_wt = io.network_structure_from_nx(
    #     G_nx_dual, crs=edges_gdf.crs
    # )
    # # compute length weighted centralities
    # nodes_gdf = networks.node_centrality_shortest(
    #     network_structure_len_wt, nodes_gdf, distances=cent_distances
    # )
    # # rename length weighted columns for saving (prevents overwriting)
    # for col_extract in [
    #     "cc_density",
    #     "cc_beta",
    #     "cc_farness",
    #     "cc_harmonic",
    #     "cc_hillier",
    #     "cc_betweenness",
    #     "cc_betweenness_beta",
    # ]:
    #     new_col_extract = col_extract.replace("cc_", "cc_lw_")
    #     nodes_gdf.columns = [
    #         col.replace(col_extract, new_col_extract) for col in nodes_gdf.columns
    #     ]
    # # compute unweighted centralities
    # nodes_gdf = networks.node_centrality_shortest(
    #     network_structure, nodes_gdf, distances=cent_distances
    # )
    # # compute accessibilities
    # nodes_gdf, prems = layers.compute_accessibilities(
    #     prems,
    #     landuse_column_label="division_desc",
    #     accessibility_keys=[
    #         "food_bev",
    #         "creat_entert",
    #         "retail",
    #         "services",
    #         "education",
    #         "accommod",
    #         "sports_rec",
    #         "health",
    #     ],
    #     nodes_gdf=nodes_gdf,
    #     network_structure=network_structure,
    #     distances=lu_distances,
    # )
    # # compute mixed uses
    # nodes_gdf, prems = layers.compute_mixed_uses(
    #     prems,
    #     landuse_column_label="division_desc",
    #     nodes_gdf=nodes_gdf,
    #     network_structure=network_structure,
    #     distances=lu_distances,
    # )
    # # filter by live nodes
    nodes_gdf = nodes_gdf[nodes_gdf.live]
    # save and track in dict
    nodes_gdf.to_file(f"temp/df_{dist_par}_{dist_cons}_clean.gpkg")
    dfs[dist_par] = nodes_gdf

# %%
# load if necessary - very slow
dfs_load = {}
clean_dists = [0, 2, 4, 6, 8, 10, 12, 16, 20, 24]
if False:
    for dist in clean_dists:
        dfs_load[clean_dists] = gpd.read_file(f"temp/df_{dist}_clean.gpkg")
# manually run dfs = dfs_load to prevent accidental loss of loaded data

# %%
mad_gpd = gpd.read_file("temp/dataset.gpkg")
dfs["mn"] = mad_gpd

# %%
stats = pd.DataFrame()
for dist, df in dfs.items():
    stats.loc[dist, "node_count"] = len(df)
    for cent_dist in cent_distances:
        stats.loc[dist, f"node_density_{cent_dist}"] = np.nanmean(df[[f"cc_density_{cent_dist}"]])
        stats.loc[dist, f"street_density_{cent_dist}"] = np.nanmean(
            df[[f"cc_lw_density_{cent_dist}"]]
        )
        corr_coef, p_value = spearmanr(df[[f"cc_harmonic_{cent_dist}"]], df[["cc_food_bev_100_wt"]])
        stats.loc[dist, f"lu_corr_{cent_dist}"] = corr_coef
stats

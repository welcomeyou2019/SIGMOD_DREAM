from models import NetHSP_GIN, NetGIN, NetGCN, NetGAT

def get_model(args, num_features=None, num_classes=None):
    if args.second_model == "GIN":
        emb_sizes = [args.emb_dim] * (args.num_layers + 1)
        if args.dataset_name.endswith("Prox"):
            emb_input = 10
        else:
            emb_input = -1
        model = NetGIN(
            num_features,
            num_classes,
            emb_sizes=emb_sizes,
            device=args.device,
            scatter=args.scatter,
            drpt_prob=args.dropout,
            eps=args.eps,
            train_eps=False,
            emb_input=emb_input,
        )
        return model
    elif args.second_model == "GIN-EPS":
        if args.dataset_name.endswith("Prox"):
            emb_input = 10  # The number of colors in the proximity dataset
        else:
            emb_input = -1
        emb_sizes = [args.emb_dim] * (args.num_layers + 1)
        model = NetGIN(
            num_features,
            num_classes,
            emb_sizes=emb_sizes,
            device=args.device,
            scatter=args.scatter,
            drpt_prob=args.dropout,
            eps=args.eps,
            train_eps=True,
            emb_input=emb_input,
        )
        return model
    elif args.second_model == "GCN":
        if args.dataset_name.endswith("Prox"):
            emb_input = 10
        else:
            emb_input = -1
        emb_sizes = [args.emb_dim] * (args.num_layers + 1)
        model = NetGCN(
            num_features,
            num_classes,
            emb_sizes=emb_sizes,
            device=args.device,
            scatter=args.scatter,
            drpt_prob=args.dropout,
            emb_input=emb_input,
        )
        return model

    elif args.second_model == "GAT":
        if args.dataset_name.endswith("Prox"):
            emb_input = 10
        else:
            emb_input = -1
        emb_sizes = [args.emb_dim] * (args.num_layers + 1)
        model = NetGAT(
            num_features,
            num_classes,
            emb_sizes=emb_sizes,
            device=args.device,
            scatter=args.scatter,
            drpt_prob=args.dropout,
            emb_input=emb_input,
        )
        return model

    inside, outside = args.second_model.lower().split("-")[1:3]

    if not inside in ["attn_nh", "global_attn_nh", "sum", "rsum", "edgesum"]:
        raise ValueError("Invalid inside model aggregation.")

    if not outside in ["sum", "weight", "eps_weight"]:
        raise ValueError("Invalid outside model aggregation.")

    emb_sizes = [args.emb_dim] * (args.num_layers + 1)

    model = NetHSP_GIN(
        num_features,
        num_classes,
        emb_sizes=emb_sizes,
        device=args.device,
        max_distance=args.max_distance,
        scatter=args.scatter,
        drpt_prob=args.dropout,
        inside_aggr=inside,
        outside_aggr=outside,
        mode=args.second_model,
        eps=args.eps,
        batch_norm=args.batch_norm,
        layer_norm=args.layer_norm,
        pool_gc=args.pool_gc,
        residual_frequency=args.res_freq,
        dataset=args.dataset_name,
        learnable_emb=args.learnable_emb,
        use_feat=args.use_feat
    ).to(args.device)
    return model

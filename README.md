# ComFy
Community+Feature Similarity based Graph Rewiring

Requirements - 

```Python
1. Pytorch = 2.2.1
2. Pytorch-Geometric - 2.5.2
3. DGL - 2.2.1+cu121
```

ComFy has following directories - 

1. Comm+Sim - this has all the code need to reproduce results for rewiring based on maximizing the feature similarity (FeaSt) and Community detection plus feature similarity based rewiring (ComFy).

2. CommunityRewiring - contains code to delete/add inter-class edges/intra-class edges directly based on communities detected.

3. Spectral - contains codes to maximize/minimise spectral gap and add/delete edges.

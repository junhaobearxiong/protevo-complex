# protevo-complex
Modeling evolution of protein complexes p(y2, x2 | y1, x1, ty, tx), where x and y are proteins in a protein complex.

## Model optimization
- [ ] Multi GPU training
- [ ] Currently, both `attn_mask_in_length` and `distance_tensor` are of shape (B, L)
    which is wasteful since there are only 2*B nonzero elements and L is often large 500-1k.
    Changing this requires changing all the flash attention operations

## Backburner
- [ ] The original dataset from the HumanPPI paper doesn't seem to include insertion relative to the query sequence (human).
    Also, it is unclear what the lower case characters mean.
- [ ] Faster / more memory efficient implementation of sequence identity filter of the subsampled MSA
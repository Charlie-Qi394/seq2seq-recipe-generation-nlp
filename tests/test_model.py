import torch

from recipe_generation.model import ModelConfig, Seq2Seq


def test_seq2seq_forward_shape() -> None:
    config = ModelConfig(
        source_vocab_size=12,
        target_vocab_size=15,
        embedding_dim=8,
        hidden_size=8,
        use_attention=True,
    )
    model = Seq2Seq(config)
    src = torch.tensor([[1, 4, 5, 2], [1, 6, 2, 0]], dtype=torch.long)
    src_lengths = torch.tensor([4, 3], dtype=torch.long)
    tgt = torch.tensor([[1, 4, 5, 2], [1, 7, 2, 0]], dtype=torch.long)

    outputs = model(src, src_lengths, tgt)

    assert outputs.shape == (2, 4, 15)

import torch
from datamodule.transforms import TextTransform
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
 
 
def get_beam_search_decoder(
    model,
    token_list,
    penalty=0.0,
    ctc_weight=0.1,
    beam_size=40,
):
    """Build the BatchBeamSearch decoder."""
    sos = model.odim - 1
    eos = model.odim - 1
    scorers = model.scorers()
    scorers["length_bonus"] = LengthBonus(len(token_list))
    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "length_bonus": penalty,
    }
    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=sos,
        eos=eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )
 
 
class InferenceModule(torch.nn.Module):
    """Visual-only inference wrapper around the Auto-AVSR E2E model."""
 
    def __init__(
        self,
        modality,
        ckpt_path,
        beam_size=40,
        ctc_weight=0.1,
        penalty=0.0,
    ):
        super().__init__()
        self.modality = modality
        self.beam_size = beam_size
        self.ctc_weight = ctc_weight
        self.penalty = penalty
        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list
        self.model = E2E(
            odim=len(self.token_list),
            modality=modality,
            ctc_weight=ctc_weight,
        )
 
        # Load visual model checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = {
                k.replace("model.", "", 1): v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("model.")
            }
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            missing, unexpected = self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            missing, unexpected = self.model.load_state_dict(ckpt, strict=False)
 
        if missing:
            print(f"[warn] {len(missing)} missing keys, first 5: {missing[:5]}")
        if unexpected:
            print(f"[warn] {len(unexpected)} unexpected keys, first 5: {unexpected[:5]}")
        if not missing and not unexpected:
            print("[ok] Visual checkpoint loaded cleanly (no missing or unexpected keys).")
 
        # Beam search is built lazily on first access (see the property below).
        self._beam_search = None
 
    @property
    def beam_search(self):
        """Lazily build the beam search decoder on first access."""
        if self._beam_search is None:
            self._beam_search = get_beam_search_decoder(
                self.model,
                self.token_list,
                penalty=self.penalty,
                ctc_weight=self.ctc_weight,
                beam_size=self.beam_size,
            )
        return self._beam_search
 
    @torch.no_grad()
    def forward(self, video):
        """Run inference on a single pre-processed video.
 
        Args:
            video: tensor of shape (T, 1, 88, 88).
        Returns:
            str: predicted transcript (uppercase English).
        """
        x = self.model.frontend(video.unsqueeze(0))
        x = self.model.proj_encoder(x)
        enc_feat, _ = self.model.encoder(x, None)
        enc_feat = enc_feat.squeeze(0)
        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[:1]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        return predicted
 
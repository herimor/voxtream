from voxtream.utils.aligner.charsiu import CharsiuForcedAligner
from voxtream.utils.aligner.clap_ipa import ClapIPA
from voxtream.utils.aligner.phoneme_aligner import PhonemeAligner as PhonemeAligner

ALIGNERS = {
    "charsiu/en_w2v2_fc_10ms": CharsiuForcedAligner,
    "clap_ipa": ClapIPA,
}

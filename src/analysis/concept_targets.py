"""Canonical concept → designed target class mapping for HAM10000.

Each entry maps a concept_id (as it appears in ham10000_prompts_*.txt and the
``concept_ids`` array in every .npz file) to the HAM10000 diagnostic class it
was designed to detect.

Rationale for each group
-------------------------
mel   : ABCD rule + 7-point checklist concepts are canonically defined for
        melanoma detection in the dermatoscopy literature.
nv    : symmetric_uniform and reticular_network describe benign melanocytic
        nevus appearance; healthy_skin is treated as a benign/nv-aligned
        reference (absence of lesion markers).
bcc   : arborizing vessels, blue-grey ovoid nests, and ulceration are the
        three canonical BCC-specific dermoscopic criteria.
akiec : scaly surface and strawberry pseudonetwork are the two main AKIEC
        (actinic keratosis / Bowen's disease) dermoscopic features.
bkl   : milia-like cysts, comedo-like openings, and cerebriform surface are
        the classic seborrhoeic keratosis (bkl) criteria.
df    : central white patch (scar-like area) is the primary dermatofibroma
        dermoscopic criterion.
vasc  : red lacunae define vascular lesions (haemangioma / angiokeratoma).
"""

from types import MappingProxyType

CONCEPT_TARGET_CLASS: dict[str, str] = MappingProxyType({  # type: ignore[assignment]
    # Melanoma-targeted (ABCD rule)
    "asymmetry":                 "mel",
    "border_irregularity":       "mel",
    "colour_variation":          "mel",
    "diameter_large":            "mel",
    # Melanoma-targeted (7-point checklist)
    "atypical_pigment_network":  "mel",
    "blue_white_veil":           "mel",
    "atypical_vascular_pattern": "mel",
    "irregular_streaks":         "mel",
    "irregular_pigmentation":    "mel",
    "irregular_dots_globules":   "mel",
    "regression_structures":     "mel",
    # Benign nevus anchors
    "symmetric_uniform":         "nv",
    "reticular_network":         "nv",
    # BCC-targeted
    "arborizing_vessels":        "bcc",
    "blue_grey_ovoid_nests":     "bcc",
    "ulceration":                "bcc",
    # AKIEC-targeted
    "scaly_surface":             "akiec",
    "strawberry_pattern":        "akiec",
    # BKL-targeted
    "milia_like_cysts":          "bkl",
    "comedo_like_openings":      "bkl",
    "cerebriform_surface":       "bkl",
    # DF-targeted
    "central_white_patch":       "df",
    # Vascular
    "red_lacunae":               "vasc",
    # Benign reference (no lesion marker; treated as nv-aligned)
    "healthy_skin":              "nv",
})

# Malignant disease groups for the "malignant-only" summary statistic
MALIGNANT_CLASSES = frozenset({"mel", "bcc", "akiec"})

# Figure : boucle de raffinement du prompt (question 1, acquisition des données)
# Adapté de la documentation officielle de Matplotlib, tutoriel « Annotations »
#   https://matplotlib.org/stable/users/explain/text/annotations.html
#   (sections « Annotating with boxed text » et « Coordinate systems for annotations » :
#    texte encadré avec bbox=dict(boxstyle=...), flèches avec annotate(arrowprops=...),
#    et un Text utilisé comme système de coordonnées pour relier deux boîtes)
# Adaptation au projet : Claude (Anthropic), déclarée comme aide de l'IA.

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Couleurs par type d'étape
COULEURS = {
    "donnees": "#e0e0e0",  # documents et prompts
    "qwen": "#c6dbef",     # étapes faites par Qwen
    "humain": "#fdd0a2",   # validations humaines
    "test": "#c7e9c0",     # étape automatique
    "final": "#74c476",    # résultat
}

fig, ax = plt.subplots(figsize=(15, 5.5))
ax.set_xlim(0, 17)
ax.set_ylim(0.3, 6.4)
ax.axis("off")


def boite(x, y, texte, type_etape, gras=False):
    return ax.text(
        x, y, texte, ha="center", va="center", fontsize=10.5,
        fontweight="bold" if gras else "normal", wrap=True,
        bbox=dict(boxstyle="round,pad=0.6", fc=COULEURS[type_etape], ec="#444444", lw=1.2),
    )


def fleche(source, cible, depart=(1, 0.5), arrivee=(0, 0.5), couleur="#333333"):
    ax.annotate(
        "", xy=arrivee, xycoords=cible, xytext=depart, textcoords=source,
        arrowprops=dict(arrowstyle="-|>", color=couleur, lw=1.6, shrinkA=9, shrinkB=9,
                        connectionstyle="arc3,rad=0"),
    )


# En amont de la boucle
doc = boite(1.3, 5.6, "Documentation\nde l'API LNH\n(RAG)", "donnees")
prompt0 = boite(3.9, 5.6, "Prompt initial\n", "donnees")

#La boucle (sens horaire)
gen_code = boite(6.9, 5.6, "1. Génération\ndu code", "qwen")
valid_code = boite(9.7, 5.6, "2. Validation\nhumaine\ndu code", "humain")
tests = boite(12.5, 5.6, "3. Passage\ndes tests", "test")
decision = boite(12.5, 3.3, "Tous les tests\npassent ?", "test")
decision.get_bbox_patch().set_boxstyle("round4,pad=0.6")
mise_commun = boite(12.5, 1.1, "4. Mise en commun :\ncode + prompt + résultats\ndes tests", "donnees")
gen_prompt = boite(9.7, 1.1, "5. Génération\nd'un nouveau\nprompt", "qwen")
valid_prompt = boite(6.9, 1.1, "6. Validation\nhumaine\ndu prompt", "humain")

# En aval
final = boite(15.6, 3.3, "Code final", "final", gras=True)

# Flèches
fleche(doc, prompt0)
fleche(prompt0, gen_code)
fleche(gen_code, valid_code)
fleche(valid_code, tests)
fleche(tests, decision, depart=(0.5, 0), arrivee=(0.5, 1))
fleche(decision, final, couleur="#238b45")
ax.text(14.3, 3.55, "Oui", ha="center", fontsize=11, fontweight="bold", color="#238b45")
fleche(decision, mise_commun, depart=(0.5, 0), arrivee=(0.5, 1), couleur="#cb181d")
ax.text(12.85, 2.2, "Non", ha="left", fontsize=11, fontweight="bold", color="#cb181d")
fleche(mise_commun, gen_prompt, depart=(0, 0.5), arrivee=(1, 0.5))
fleche(gen_prompt, valid_prompt, depart=(0, 0.5), arrivee=(1, 0.5))
fleche(valid_prompt, gen_code, depart=(0.5, 1), arrivee=(0.5, 0))

# Étiquette au centre de la boucle
ax.text(9.7, 3.3, "Boucle de raffinement\ndu prompt", ha="center", va="center",
        fontsize=12, style="italic", color="#555555")

# Légende des types d'étapes
legende = [
    Patch(fc=COULEURS["qwen"], ec="#444444", label="Qwen3-4B"),
    Patch(fc=COULEURS["humain"], ec="#444444", label="Humain"),
    Patch(fc=COULEURS["test"], ec="#444444", label="Automatique (tests)"),
    Patch(fc=COULEURS["donnees"], ec="#444444", label="Documents / prompts"),
]
ax.legend(handles=legende, loc="lower left", bbox_to_anchor=(0.0, 0.18), fontsize=10, frameon=False)

ax.set_title("Génération du code d'acquisition avec Qwen + RAG : boucle de raffinement du prompt",
             fontsize=13, pad=12)
fig.tight_layout()
fig.savefig("boucle_raffinement.png", dpi=200, bbox_inches="tight")
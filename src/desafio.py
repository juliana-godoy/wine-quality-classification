"""
Tech Challenge - Fase 02
Classificacao da Qualidade de Vinhos com Machine Learning

Objetivo: prever se um vinho é de Alta Qualidade (quality >= 7) ou
Baixa/Media Qualidade (quality < 7) a partir de caracteristicas fisico-quimicas.

COMO RODAR:
    1. Execute:  python src/desafio.py
    2. Todos os graficos, tabelas (.csv), o modelo final (.joblib) e um
       relatorio de texto será salvo na pasta 'results/'.
"""

import io
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Usa backend nao-interativo para que o script rode por completo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

import joblib



# 1. CONFIGURACOES GERAIS

RANDOM_STATE = 42

# Diretorio base = raiz do projeto (pasta acima de src/).
BASE_DIR = Path(__file__).resolve().parent.parent

# Estrutura de pastas de saida.
OUTPUT_DIR    = BASE_DIR / "results"
DIR_GRAFICOS  = OUTPUT_DIR / "graficos"
DIR_TABELAS   = OUTPUT_DIR / "tabelas"
DIR_RELATORIOS = OUTPUT_DIR / "relatorios"

for pasta in (OUTPUT_DIR, DIR_GRAFICOS, DIR_TABELAS, DIR_RELATORIOS):
    pasta.mkdir(exist_ok=True)



# Relatorio de texto: Tudo o que é impresso/exibido
# permanece gravado em results/relatorios/relatorio_analise.txt

_RELATORIO_PATH = DIR_RELATORIOS / "relatorio_analise.txt"
_relatorio_buffer = []


def log(*args, sep=" "):
    """Imprime na tela e acumula no relatorio de texto."""
    texto = sep.join(str(a) for a in args)
    print(texto)
    _relatorio_buffer.append(texto)


def salvar_relatorio():
    """Grava o relatorio acumulado em arquivo de texto."""
    _RELATORIO_PATH.write_text("\n".join(_relatorio_buffer), encoding="utf-8")
    print(f"\nRelatorio de texto salvo em: {_RELATORIO_PATH}")



# 2. PADRAO VISUAL DOS GRAFICOS

COR_PRINCIPAL = "#5A0F1B"      # Vinho escuro principal
COR_SECUNDARIA = "#9B1B30"     # Vinho medio
COR_CLARA = "#E8B4BC"          # Vinho claro
COR_GRID = "#D9D9D9"           # Cinza claro para linhas de fundo
COR_TEXTO = "#000000"          # Preto para texto
COR_FUNDO = "#FFFFFF"          # Branco para fundo

# Gradiente utilizado nos graficos
CMAP_VERDE = LinearSegmentedColormap.from_list(
    "gradiente_verde",
    [COR_CLARA, COR_SECUNDARIA, COR_PRINCIPAL],
)

plt.rcParams.update({
    "figure.facecolor": COR_FUNDO,
    "axes.facecolor": COR_FUNDO,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "font.weight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.edgecolor": COR_TEXTO,
    "axes.labelcolor": COR_TEXTO,
    "xtick.color": COR_TEXTO,
    "ytick.color": COR_TEXTO,
    "legend.fontsize": 10,
    "legend.frameon": True,
})


def gerar_cores_gradiente(qtd, cor_inicio=COR_PRINCIPAL, cor_fim=COR_CLARA):
    """Gera uma lista de cores em gradiente."""
    cmap = LinearSegmentedColormap.from_list("gradiente_custom", [cor_inicio, cor_fim])
    return [cmap(i / max(qtd - 1, 1)) for i in range(qtd)]


def formatar_grafico(ax, titulo=None, xlabel=None, ylabel=None, grid_axis="y", usar_grid=True):
    """Aplica o padrao visual: titulos, eixos e numeros em negrito + grid."""
    if titulo:
        ax.set_title(titulo, fontweight="bold", color=COR_TEXTO, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontweight="bold", color=COR_TEXTO)
    if ylabel:
        ax.set_ylabel(ylabel, fontweight="bold", color=COR_TEXTO)

    ax.set_axisbelow(True)

    if usar_grid:
        ax.grid(True, axis=grid_axis, linestyle="-", linewidth=0.8, alpha=0.65, color=COR_GRID)

    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
        label.set_color(COR_TEXTO)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_color(COR_TEXTO)
    for spine in ax.spines.values():
        spine.set_color(COR_TEXTO)
        spine.set_linewidth(0.8)

    return ax


def adicionar_rotulos_barras(ax, orientacao="vertical", prefixo="", sufixo="", casas_decimais=0):
    """Adiciona valores nas barras em negrito (vertical ou horizontal)."""
    for barra in ax.patches:
        if orientacao == "horizontal":
            valor = barra.get_width()
            x = valor
            y = barra.get_y() + barra.get_height() / 2
            texto = f"{prefixo}{valor:,.{casas_decimais}f}{sufixo}"
            ax.annotate(texto, xy=(x, y), xytext=(6, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=9, fontweight="bold", color=COR_TEXTO)
        else:
            valor = barra.get_height()
            x = barra.get_x() + barra.get_width() / 2
            y = valor
            texto = f"{prefixo}{valor:,.{casas_decimais}f}{sufixo}"
            ax.annotate(texto, xy=(x, y), xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold", color=COR_TEXTO)


def salvar_figura(fig, nome_arquivo):
    """Salva a figura em results/graficos/ e fecha para liberar memoria."""
    caminho = DIR_GRAFICOS / nome_arquivo
    fig.savefig(caminho, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> grafico salvo: graficos/{caminho.name}")



# 3. CARREGAMENTO DA BASE WineQT.csv

def localizar_arquivo_csv():
    """Procura o WineQT.csv em caminhos comuns (local e Colab)."""
    caminhos_possiveis = [
        BASE_DIR / "data" / "WineQT.csv",
        BASE_DIR / "WineQT.csv",
        Path("data/WineQT.csv"),
        Path("WineQT.csv"),
        Path("../data/WineQT.csv"),
        Path("/content/WineQT.csv"),
        Path("/content/data/WineQT.csv"),
        Path("/mnt/data/WineQT.csv"),
    ]
    for caminho in caminhos_possiveis:
        if caminho.exists():
            return caminho

    raise FileNotFoundError(
        "Nao encontrei o WineQT.csv.\n"
        f"Coloque o arquivo 'WineQT.csv' na pasta:\n   {BASE_DIR / 'data'}\n"
        "e rode novamente."
    )


def main():
    log("=" * 80)
    log("TECH CHALLENGE FASE 02 - CLASSIFICACAO DA QUALIDADE DE VINHOS")
    log("Execucao iniciada em:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log("=" * 80)

    DATA_PATH = localizar_arquivo_csv()
    log(f"\nArquivo utilizado: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    log("Dimensao da base:", df.shape)
    log("\nPrimeiras linhas:")
    log(df.head().to_string())


    # 4. CRIACAO DA VARIAVEL ALVO

    df["quality_class"] = (df["quality"] >= 7).astype(int)
    df["quality_label"] = np.where(df["quality_class"] == 1, "Alta Qualidade", "Baixa/Media Qualidade")

    log("\n" + "=" * 80)
    log("4. VARIAVEL ALVO (quality_class: 1 = Alta, 0 = Baixa/Media)")
    log("=" * 80)
    log(df[["quality", "quality_class", "quality_label"]].head(10).to_string())


    # 5. VISAO GERAL E QUALIDADE DOS DADOS

    log("\n" + "=" * 80)
    log("5. VISAO GERAL E QUALIDADE DOS DADOS")
    log("=" * 80)

    log("\nInformacoes gerais:")
    info_buffer = io.StringIO()
    df.info(buf=info_buffer)
    log(info_buffer.getvalue())

    log("\nTotal de valores faltantes por coluna:")
    log(df.isna().sum().to_string())

    log("\nTotal de registros duplicados:", df.duplicated().sum())

    log("\nEstatisticas descritivas:")
    log(df.describe().T.to_string())


    # 6. ANALISE EXPLORATORIA - EDA

    log("\n" + "=" * 80)
    log("6. ANALISE EXPLORATORIA DE DADOS (EDA)")
    log("=" * 80)

    # 6.1 Distribuicao da nota original de qualidade
    log("\n6.1 Distribuicao das notas de qualidade")
    quality_counts = df["quality"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    cores = gerar_cores_gradiente(len(quality_counts))
    ax.bar(quality_counts.index.astype(str), quality_counts.values,
           color=cores, edgecolor=COR_TEXTO, linewidth=0.4)
    formatar_grafico(ax, titulo="Distribuicao das Notas de Qualidade",
                     xlabel="Nota de Qualidade", ylabel="Quantidade de Amostras", grid_axis="y")
    adicionar_rotulos_barras(ax, orientacao="vertical", casas_decimais=0)
    fig.tight_layout()
    salvar_figura(fig, "distribuicao_quality.png")

    # 6.2 Balanceamento das classes
    log("\n6.2 Balanceamento das classes")
    class_counts = df["quality_label"].value_counts().reindex(["Baixa/Media Qualidade", "Alta Qualidade"])
    class_percent = (class_counts / class_counts.sum() * 100).round(2)
    balanceamento = pd.DataFrame({"quantidade": class_counts, "percentual": class_percent})
    log(balanceamento.to_string())

    fig, ax = plt.subplots(figsize=(10, 6))
    cores = gerar_cores_gradiente(len(class_counts))
    ax.bar(class_counts.index, class_counts.values, color=cores, edgecolor=COR_TEXTO, linewidth=0.4)
    formatar_grafico(ax, titulo="Balanceamento das Classes",
                     xlabel="Classe", ylabel="Quantidade", grid_axis="y")
    adicionar_rotulos_barras(ax, orientacao="vertical", casas_decimais=0)
    for label in ax.get_xticklabels():
        label.set_rotation(10)
        label.set_fontweight("bold")
    fig.tight_layout()
    salvar_figura(fig, "balanceamento_classes.png")

    # 6.3 Correlacoes entre variaveis
    log("\n6.3 Correlacoes entre variaveis")
    corr_df = df.drop(columns=["Id", "quality_label"], errors="ignore").corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(corr_df.values, vmin=-1, vmax=1, cmap=CMAP_VERDE)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlacao", fontweight="bold", color=COR_TEXTO)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_color(COR_TEXTO)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=90)
    ax.set_yticklabels(corr_df.index)
    for i in range(len(corr_df.index)):
        for j in range(len(corr_df.columns)):
            valor = corr_df.iloc[i, j]
            cor_texto = "#FFFFFF" if valor > 0.55 else COR_TEXTO
            ax.text(j, i, f"{valor:.2f}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color=cor_texto)
    formatar_grafico(ax, titulo="Matriz de Correlacao entre Variaveis",
                     xlabel="Variaveis", ylabel="Variaveis", usar_grid=False)
    fig.tight_layout()
    salvar_figura(fig, "matriz_correlacao.png")

    correlacao_quality = (
        corr_df["quality"].drop("quality", errors="ignore")
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    log("\nCorrelacao de cada variavel com 'quality':")
    log(correlacao_quality.to_frame("correlacao_com_quality").to_string())

    log("\nInterpretacao das correlacoes com a variavel alvo (quality):")
    log("-" * 60)
    log("alcohol        (+0.48): Correlacao positiva mais forte entre as features.")
    log("  O teor alcoolico e produto direto da fermentacao. Fermentacoes")
    log("  bem conduzidas convertem mais acucar em alcool, gerando vinhos")
    log("  mais encorpados, o que os avaliadores tendem a pontuar mais alto.")
    log("")
    log("volatile acidity (-0.41): Segunda maior correlacao em modulo, negativa.")
    log("  A acidez volatil e composta principalmente de acido acetico,")
    log("  produzido por bacterias durante fermentacao irregular ou oxidacao.")
    log("  Niveis elevados conferem sabor de vinagre, deteriorando a qualidade.")
    log("")
    log("sulphates      (+0.26): Correlacao positiva moderada.")
    log("  Os sulfatos atuam como antimicrobianos e antioxidantes. Niveis")
    log("  adequados conferem estabilidade e frescor ao vinho.")
    log("")
    log("citric acid    (+0.24): Correlacao positiva moderada.")
    log("  Contribui para frescor e acidez equilibrada. Presente em")
    log("  niveis moderados, e associado a um perfil de sabor mais agradavel.")
    log("")
    log("total sulfur dioxide (-0.18): Correlacao negativa fraca-moderada.")
    log("  Embora necessario como conservante, excesso de SO2 total")
    log("  pode ser perceptivel no aroma e sabor, penalizando a nota.")
    log("")
    log("density        (-0.18): Correlacao negativa fraca-moderada.")
    log("  A densidade e inversamente proporcional ao teor alcoolico:")
    log("  quanto mais acucar e convertido em alcool, menor a densidade.")
    log("  Essa variavel reflete indiretamente o mesmo fenomeno que alcohol.")
    log("")
    log("chlorides      (-0.12): Correlacao negativa fraca.")
    log("  Niveis elevados de cloreto (sal) podem desequilibrar o sabor.")
    log("  A correlacao fraca indica que, no intervalo deste dataset,")
    log("  o impacto nao e preponderante frente a outras variaveis.")
    log("")
    log("fixed acidity  (+0.12): Correlacao positiva fraca.")
    log("  Contribui para a estrutura e frescor, mas a correlacao fraca")
    log("  indica que o nivel absoluto importa menos do que o equilibrio")
    log("  com os demais acidos presentes no vinho.")
    log("")
    log("free sulfur dioxide (-0.06): Correlacao negativa muito fraca.")
    log("  O SO2 livre tem funcao protetora contra oxidacao e bacterias,")
    log("  mas sua correlacao linear com a qualidade e quase nula,")
    log("  indicando que a fracao livre isoladamente nao determina a nota.")
    log("")
    log("pH             (-0.05): Correlacao negativa muito fraca.")
    log("  Mede a acidez total, mas sua correlacao proxima de zero indica")
    log("  que o tipo e composicao dos acidos (acidez fixa, volatil e")
    log("  citrica) importam mais do que o pH geral como indicador.")
    log("")
    log("residual sugar (+0.02): Correlacao praticamente nula.")
    log("  Para este dataset de vinhos tintos, o acucar residual nao")
    log("  apresenta relacao linear com a qualidade percebida pelos")
    log("  avaliadores. Os vinhos se concentram em um perfil seco onde")
    log("  o acucar nao e fator diferenciador de nota.")

    # 6.4 Outliers pelo metodo IQR
    log("\n6.4 Outliers pelo metodo IQR")

    def detectar_outliers_iqr(dataframe, colunas):
        linhas = []
        for col in colunas:
            q1 = dataframe[col].quantile(0.25)
            q3 = dataframe[col].quantile(0.75)
            iqr = q3 - q1
            limite_inferior = q1 - 1.5 * iqr
            limite_superior = q3 + 1.5 * iqr
            qtd_outliers = ((dataframe[col] < limite_inferior) | (dataframe[col] > limite_superior)).sum()
            linhas.append({
                "variavel": col,
                "limite_inferior": limite_inferior,
                "limite_superior": limite_superior,
                "qtd_outliers": int(qtd_outliers),
                "perc_outliers": round(qtd_outliers / len(dataframe) * 100, 2),
            })
        return pd.DataFrame(linhas).sort_values("qtd_outliers", ascending=False)

    colunas_numericas = [
        col for col in df.select_dtypes(include=np.number).columns
        if col not in ["Id", "quality", "quality_class"]
    ]
    outliers = detectar_outliers_iqr(df, colunas_numericas)
    log(outliers.to_string(index=False))
    outliers.to_csv(DIR_TABELAS / "outliers_iqr.csv", index=False)
    print(f"  -> tabela salva: tabelas/outliers_iqr.csv")

    log("\nDecisao sobre tratamento de outliers:")
    log("-" * 60)
    log("Os outliers detectados foram MANTIDOS no dataset. Justificativa:")
    log("")
    log("1. Natureza dos dados: as variaveis sao medicoes fisico-quimicas")
    log("   reais de amostras de vinho. Valores extremos representam")
    log("   vinhos genuinamente atipicos, nao erros de medicao ou registro.")
    log("")
    log("2. Percentuais controlados: a maioria das variaveis apresenta")
    log("   menos de 4% de outliers. As mais afetadas sao residual sugar")
    log("   (9.62%, 110 amostras) e chlorides (6.74%, 77 amostras),")
    log("   que ainda assim representam valores fisiologicamente possiveis.")
    log("")
    log("3. Robustez do modelo: o Random Forest, melhor modelo obtido,")
    log("   e baseado em arvores de decisao que realizam divisoes por")
    log("   threshold. Esse mecanismo torna o algoritmo naturalmente")
    log("   robusto a valores extremos sem necessidade de remocao previa.")
    log("")
    log("4. Preservacao da classe minoritaria: o dataset possui apenas")
    log("   159 amostras de Alta Qualidade (13.91% do total). Remover")
    log("   outliers sem criterio rigoroso poderia eliminar exatamente")
    log("   os vinhos excepcionais mais informativos para o classificador.")

    # 6.5 Comparacao de variaveis por classe (boxplots)
    log("\n6.5 Boxplots por classe de qualidade")
    variaveis_para_boxplot = ["alcohol", "volatile acidity", "sulphates", "citric acid"]
    for col in variaveis_para_boxplot:
        if col not in df.columns:
            log(f"  (aviso) coluna '{col}' nao encontrada, boxplot ignorado.")
            continue
        dados_plot = [
            df.loc[df["quality_class"] == 0, col].values,
            df.loc[df["quality_class"] == 1, col].values,
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        box = ax.boxplot(
            dados_plot,
            tick_labels=["Baixa/Media Qualidade", "Alta Qualidade"],
            patch_artist=True,
            medianprops={"color": COR_TEXTO, "linewidth": 1.5},
            boxprops={"linewidth": 1.2, "color": COR_TEXTO},
            whiskerprops={"linewidth": 1.2, "color": COR_TEXTO},
            capprops={"linewidth": 1.2, "color": COR_TEXTO},
            flierprops={"marker": "o", "markerfacecolor": COR_SECUNDARIA,
                        "markeredgecolor": COR_TEXTO, "markersize": 5, "alpha": 0.75},
        )
        cores = gerar_cores_gradiente(len(box["boxes"]))
        for patch, cor in zip(box["boxes"], cores):
            patch.set_facecolor(cor)
        formatar_grafico(ax, titulo=f"{col} por Classe de Qualidade",
                         xlabel="Classe", ylabel=col, grid_axis="y")
        for label in ax.get_xticklabels():
            label.set_rotation(15)
            label.set_fontweight("bold")
        fig.tight_layout()
        salvar_figura(fig, f"boxplot_{col.replace(' ', '_')}_por_classe.png")


    # 7. PRE-PROCESSAMENTO

    log("\n" + "=" * 80)
    log("7. PRE-PROCESSAMENTO DOS DADOS")
    log("=" * 80)

    feature_cols = [
        col for col in df.columns
        if col not in ["Id", "quality", "quality_class", "quality_label"]
    ]

    log("\nFeature Engineering:")
    log("-" * 60)
    log("Nao foram criadas novas features. Avaliacao realizada:")
    log("")
    log("1. As 11 variaveis originais representam diretamente as")
    log("   propriedades fisico-quimicas do vinho com significado")
    log("   interpretavel: alcohol, volatile acidity, sulphates,")
    log("   citric acid, density, chlorides, pH, residual sugar,")
    log("   fixed acidity, free e total sulfur dioxide.")
    log("")
    log("2. A analise de correlacao mostrou que alcohol (+0.48) e")
    log("   volatile acidity (-0.41) ja capturam os sinais mais fortes")
    log("   de qualidade de forma direta, sem necessidade de combinacoes.")
    log("")
    log("3. density e matematicamente relacionada a alcohol (densidade")
    log("   diminui conforme o teor alcoolico sobe). Optou-se por manter")
    log("   ambas, pois o Random Forest lida com multicolinearidade")
    log("   sem degradacao de desempenho, e cada variavel contribui")
    log("   com informacao independente nos splits das arvores.")
    log("")
    log("4. O dataset possui 1143 registros e classe minoritaria com")
    log("   apenas 159 amostras (13.91%). Criar features polinomiais")
    log("   ou de interacao ampliaria o espaco de atributos com risco")
    log("   real de overfitting para a classe de Alta Qualidade.")
    log("")
    log("Conclusao: o conjunto original de 11 features foi considerado")
    log("suficiente e adequado para o problema proposto.")

    X = df[feature_cols]
    y = df["quality_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    log("Tamanho treino:", X_train.shape)
    log("Tamanho teste:", X_test.shape)
    log("\nDistribuicao no treino:")
    log(y_train.value_counts(normalize=True).rename("proporcao").to_string())
    log("\nDistribuicao no teste:")
    log(y_test.value_counts(normalize=True).rename("proporcao").to_string())


    # 8. DEFINICAO DOS MODELOS

    modelos = {
        "Baseline - Classe majoritaria": DummyClassifier(strategy="most_frequent"),
        "Regressao Logistica": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            class_weight="balanced", min_samples_leaf=2, n_jobs=-1,
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }


    # 9. AVALIACAO DOS MODELOS

    log("\n" + "=" * 80)
    log("9. TREINAMENTO E AVALIACAO DOS MODELOS")
    log("=" * 80)

    resultados = []
    relatorios = {}
    matrizes_confusao = {}
    modelos_treinados = {}

    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else None

        resultados.append({
            "modelo": nome,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_alta": precision_score(y_test, y_pred, zero_division=0),
            "recall_alta": recall_score(y_test, y_pred, zero_division=0),
            "f1_alta": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan,
        })
        relatorios[nome] = classification_report(
            y_test, y_pred, target_names=["Baixa/Media", "Alta"], zero_division=0, digits=4
        )
        matrizes_confusao[nome] = confusion_matrix(y_test, y_pred)
        modelos_treinados[nome] = modelo

    metricas_modelos = pd.DataFrame(resultados).sort_values("f1_alta", ascending=False)
    log("\nMetricas dos modelos (ordenadas por F1 da classe Alta):")
    log(metricas_modelos.to_string(index=False))
    metricas_modelos.to_csv(DIR_TABELAS / "metricas_modelos.csv", index=False)
    print("  -> tabela salva: tabelas/metricas_modelos.csv")

    for nome, relatorio in relatorios.items():
        log("\n" + "=" * 80)
        log(nome)
        log(relatorio)

    # 9.1 Matriz de confusao do melhor modelo
    melhor_modelo_nome = metricas_modelos.iloc[0]["modelo"]
    melhor_modelo = modelos_treinados[melhor_modelo_nome]
    log("\nMelhor modelo pelo F1 da classe Alta:", melhor_modelo_nome)

    y_pred_melhor = melhor_modelo.predict(X_test)
    matriz_confusao_melhor = confusion_matrix(y_test, y_pred_melhor)

    disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusao_melhor,
                                  display_labels=["Baixa/Media", "Alta"])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap=CMAP_VERDE, colorbar=False, values_format="d")
    formatar_grafico(ax, titulo=f"Matriz de Confusao - {melhor_modelo_nome}",
                     xlabel="Classe Prevista", ylabel="Classe Real", usar_grid=False)
    for texto in disp.text_.ravel():
        texto.set_fontweight("bold")
        texto.set_fontsize(13)
    fig.tight_layout()
    salvar_figura(fig, "matriz_confusao_melhor_modelo.png")

    # 9.2 Comparacao visual dos modelos
    plot_df = metricas_modelos[
        metricas_modelos["modelo"] != "Baseline - Classe majoritaria"
    ].sort_values("f1_alta")

    fig, ax = plt.subplots(figsize=(13, 6))
    cores = gerar_cores_gradiente(len(plot_df))
    ax.barh(plot_df["modelo"], plot_df["f1_alta"], color=cores, edgecolor=COR_TEXTO, linewidth=0.4)
    formatar_grafico(ax, titulo="Comparacao dos Modelos - F1 da Classe Alta Qualidade",
                     xlabel="F1-Score da Classe Alta", ylabel="Modelo", grid_axis="x")
    adicionar_rotulos_barras(ax, orientacao="horizontal", casas_decimais=3)
    fig.tight_layout()
    salvar_figura(fig, "comparacao_modelos_f1.png")


    # 10. VALIDACAO CRUZADA ESTRATIFICADA

    log("\n" + "=" * 80)
    log("10. VALIDACAO CRUZADA ESTRATIFICADA (5 folds)")
    log("=" * 80)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    linhas_cv = []
    for nome, modelo in modelos.items():
        if "Baseline" in nome:
            continue
        scores = cross_validate(modelo, X, y, cv=cv, scoring=scoring, n_jobs=1)
        linha = {"modelo": nome}
        for metrica in scoring:
            linha[f"{metrica}_mean"] = scores[f"test_{metrica}"].mean()
            linha[f"{metrica}_std"] = scores[f"test_{metrica}"].std()
        linhas_cv.append(linha)

    metricas_cv = pd.DataFrame(linhas_cv).sort_values("f1_mean", ascending=False)
    log(metricas_cv.to_string(index=False))
    metricas_cv.to_csv(DIR_TABELAS / "metricas_validacao_cruzada.csv", index=False)
    print("  -> tabela salva: tabelas/metricas_validacao_cruzada.csv")


    # 11. INTERPRETACAO - IMPORTANCIA DAS VARIAVEIS (Random Forest)

    log("\n" + "=" * 80)
    log("11. IMPORTANCIA DAS VARIAVEIS (Random Forest)")
    log("=" * 80)

    rf_model = modelos_treinados["Random Forest"]
    importancia_variaveis = (
        pd.DataFrame({"variavel": feature_cols, "importancia": rf_model.feature_importances_})
        .sort_values("importancia", ascending=False)
    )
    log(importancia_variaveis.to_string(index=False))
    importancia_variaveis.to_csv(DIR_TABELAS / "importancia_variaveis_random_forest.csv", index=False)
    print("  -> tabela salva: tabelas/importancia_variaveis_random_forest.csv")

    top_importancias = importancia_variaveis.head(10).sort_values("importancia")
    fig, ax = plt.subplots(figsize=(13, 6))
    cores = gerar_cores_gradiente(len(top_importancias))
    ax.barh(top_importancias["variavel"], top_importancias["importancia"],
            color=cores, edgecolor=COR_TEXTO, linewidth=0.4)
    formatar_grafico(ax, titulo="Top Variaveis Mais Importantes - Random Forest",
                     xlabel="Importancia", ylabel="Variavel", grid_axis="x")
    adicionar_rotulos_barras(ax, orientacao="horizontal", casas_decimais=3)
    fig.tight_layout()
    salvar_figura(fig, "importancia_variaveis.png")

    log("\nImplicacoes para o processo de producao:")
    log("-" * 60)
    log("Com base na importancia das variaveis (Random Forest) combinada")
    log("com a direcao das correlacoes com a qualidade:")
    log("")
    log("1. TEOR ALCOOLICO (importancia: 23.67% | correlacao: +0.48)")
    log("   Principal determinante de qualidade neste dataset.")
    log("   Fermentacoes completas, com controle de temperatura e saude")
    log("   das leveduras, maximizam a conversao de acucar em alcool")
    log("   e produzem vinhos mais encorpados e bem avaliados.")
    log("")
    log("2. SULFATOS (importancia: 14.60% | correlacao: +0.26)")
    log("   Segundo fator mais importante; impacto positivo na qualidade.")
    log("   Manter sulfatos na faixa observada no dataset (Q1: 0.55,")
    log("   mediana: 0.62, Q3: 0.73 g/dm3) garante protecao antioxidante")
    log("   e antimicrobiana sem exceder niveis que alterem o sabor.")
    log("")
    log("3. ACIDO CITRICO (importancia: 13.92% | correlacao: +0.24)")
    log("   Terceiro fator mais importante. Contribui para frescor e")
    log("   acidez equilibrada. Monitorar e ajustar esse parametro")
    log("   durante a producao pode elevar o perfil sensorial do vinho.")
    log("")
    log("4. ACIDEZ VOLATIL (importancia: 10.82% | correlacao: -0.41)")
    log("   Quarto fator mais importante, com impacto negativo na qualidade.")
    log("   Minimizar acidez volatil e critico: controle de higiene nas")
    log("   dornas, monitoramento de contaminacao bacteriana e dosagem")
    log("   adequada de SO2 sao as principais acoes preventivas.")
    log("")
    log("5. DIOXIDO DE ENXOFRE TOTAL (importancia: 6.05% | correlacao: -0.18)")
    log("   Correlacao negativa: excesso penaliza a nota. Usar SO2 de")
    log("   forma conservadora, suficiente para preservacao, mas evitando")
    log("   acumulo que comprometa aroma e sabor.")
    log("")
    log("6. ACUCAR RESIDUAL (importancia: 4.27% | correlacao: +0.02)")
    log("   Menor importancia entre as variaveis; correlacao praticamente")
    log("   nula. Para o perfil de vinho tinto deste dataset, o nivel")
    log("   de acucar residual nao e diferenciador de qualidade relevante.")
    log("   O foco produtivo deve estar nos fatores acima.")

    # 12. SALVANDO ARTEFATOS DO PROJETO

    log("\n" + "=" * 80)
    log("12. SALVANDO ARTEFATOS DO PROJETO")
    log("=" * 80)

    resumo_dataset = pd.DataFrame([{
        "total_registros": len(df),
        "total_colunas": df.shape[1],
        "missing_total": int(df.isna().sum().sum()),
        "duplicados": int(df.duplicated().sum()),
        "alta_qualidade": int(df["quality_class"].sum()),
        "baixa_media_qualidade": int((df["quality_class"] == 0).sum()),
        "percentual_alta": round(float(df["quality_class"].mean() * 100), 2),
    }])
    resumo_dataset.to_csv(DIR_TABELAS / "resumo_dataset.csv", index=False)

    # Modelo serializado (binario) - para RECARREGAR e fazer previsoes:
    #   import joblib; modelo = joblib.load("results/relatorios/melhor_modelo.joblib")
    joblib.dump(melhor_modelo, DIR_RELATORIOS / "melhor_modelo.joblib")

    # Resumo do melhor modelo
    # nome do algoritmo, hiperparametros e metricas no conjunto de teste.
    metricas_melhor = metricas_modelos[metricas_modelos["modelo"] == melhor_modelo_nome].iloc[0]
    linhas_resumo_modelo = [
        "=" * 70,
        "RESUMO DO MELHOR MODELO",
        "=" * 70,
        f"Algoritmo escolhido (maior F1 da classe Alta): {melhor_modelo_nome}",
        "",
        "Metricas no conjunto de teste:",
        f"  - accuracy       : {metricas_melhor['accuracy']:.4f}",
        f"  - precision_alta : {metricas_melhor['precision_alta']:.4f}",
        f"  - recall_alta    : {metricas_melhor['recall_alta']:.4f}",
        f"  - f1_alta        : {metricas_melhor['f1_alta']:.4f}",
        f"  - roc_auc        : {metricas_melhor['roc_auc']:.4f}",
        "",
        "Hiperparametros do modelo:",
    ]
    for chave, valor in melhor_modelo.get_params().items():
        linhas_resumo_modelo.append(f"  - {chave}: {valor}")
    linhas_resumo_modelo += [
        "",
        "Como reutilizar este modelo em Python:",
        "  import joblib",
        "  modelo = joblib.load('results/relatorios/melhor_modelo.joblib')",
        "  modelo.predict(novos_dados)",
    ]
    (DIR_RELATORIOS / "melhor_modelo_resumo.txt").write_text(
        "\n".join(linhas_resumo_modelo), encoding="utf-8"
    )
    print("  -> resumo legivel salvo: relatorios/melhor_modelo_resumo.txt")

    log("\nResumo do dataset:")
    log(resumo_dataset.to_string(index=False))
    log(f"\nModelo final salvo: relatorios/melhor_modelo.joblib ({melhor_modelo_nome})")
    log("Resumo legivel do modelo: relatorios/melhor_modelo_resumo.txt")
    log("\nTodos os arquivos foram salvos nas subpastas de 'results/'.")

    salvar_relatorio()
    print("\nExecucao concluida com sucesso!")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print("\n[ERRO] " + str(e), file=sys.stderr)
        sys.exit(1)

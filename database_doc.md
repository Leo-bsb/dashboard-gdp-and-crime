# **database_doc.md**

## **1. Nome do Dataset e Fonte**

**Dataset principal:** Ocorrências Criminais + PIB Municipal + População (2015–2021)
**Origem:** Banco **DataIESB**

O dataset final resulta da integração de **três tabelas originais**:

1. **Ocorrências Criminais (2015–2021)**
2. **PIB Municipal (anos correspondentes)**
3. **População – Censo 2022 (agregado por município)**

---

## **2. Contexto do Negócio**

A RIDE/DF (Região Integrada de Desenvolvimento do Distrito Federal e Entorno) reúne municípios de GO, MG e DF com forte interdependência econômica, social e de mobilidade.

A hipótese investigada:

> **O desempenho econômico dos municípios influencia os níveis de criminalidade na RIDE/DF?**

Motivações:

* municípios mais pobres podem registrar níveis mais altos de criminalidade violenta;
* regiões com PIB mais alto podem ser afetadas por crimes patrimoniais ou fluxos populacionais;
* investigar correlação entre indicadores econômicos e taxas criminais por 100 mil habitantes.

---

## **3. Problema de Pesquisa**

> **Qual a influência do desempenho econômico na criminalidade das regiões da RIDE/DF?**

---

## **4. Objetivo da Análise**

> **Explorar e identificar o comportamento da influência do desempenho econômico na criminalidade na RIDE/DF entre 2015 e 2021.**

---

## **5. Modelo Conceitual do Banco de Dados**

Apesar de o resultado final ser uma única tabela consolidada, a estrutura base é **multitabelas**, integradas da seguinte forma:

```
Ocorrências Criminais  ──┐
                         ├── (merge por código do município + ano) ───► Tabela Final
PIB Municipal ───────────┘
População (Censo) ──────────────────────────────┘
```

### **Chaves:**

* `codigo_municipio_dv_agrupado` — chave municipal padronizada (inclui caso especial de Brasília)
* `ano` — chave temporal
* População é integrada apenas por município (não tem variação anual)

---

## **6. Dicionário de Dados**

Abaixo, apenas as **colunas finais utilizadas** após tratamento. Tipos baseados no dataframe final.

| Coluna                                    | Tipo    | Descrição                                                | Exemplo        |
| ----------------------------------------- | ------- | -------------------------------------------------------- | -------------- |
| `ano`                                     | int     | Ano da observação (2015–2021)                            | 2018           |
| `codigo_municipio_dv_agrupado`            | str/int | Código IBGE agregado; Brasília tratada como único código | "5300108"      |
| `municipio_agrupado`                      | object  | Nome padronizado do município                            | "ÁGUAS LINDAS" |
| `uf`                                      | object  | Unidade Federativa                                       | "GO"           |
| `vl_agropecuaria`                         | float   | Valor adicionado bruto da agropecuária                   | 39943.78       |
| `vl_industria`                            | float   | Valor adicionado bruto da indústria                      | 9040.02        |
| `vl_servicos`                             | float   | Valor adicionado bruto de serviços                       | 72355.07       |
| `vl_administracao`                        | float   | Valor adicionado da administração pública                | 66261.04       |
| `vl_bruto_total`                          | float   | Soma dos valores agregados dos setores                   | 187599.90      |
| `vl_subsidios`                            | float   | Subsídios governamentais                                 | 10233.11       |
| `vl_pib`                                  | float   | Produto Interno Bruto do município                       | 197833.00      |
| `vl_pib_per_capta`                        | float   | PIB per capita                                           | 10857.42       |
| `Total_Habitantes`                        | int     | População total do município (Censo 2022 agregado)       | 17272          |
| `vitimas_feminicidio`                     | float   | Total anual de vítimas de feminicídio                    | 0              |
| `vitimas_homicidio_doloso`                | float   | Total anual de homicídios dolosos                        | 4              |
| `vitimas_tentativa_homicidio`             | float   | Tentativas de homicídio                                  | 3              |
| `vitimas_lesao_corporal_seguida_de_morte` | float   | Vítimas de lesão corporal seguida de morte               | 0              |
| `vitimas_transito_ou_decorrencia_dele`    | float   | Mortes no trânsito                                       | 7              |
| `vitimas_sem_indicio_de_crime`            | float   | Mortes a esclarecer                                      | 0              |
| `vitimas_latrocinio`                      | float   | Latrocínio (roubo seguido de morte)                      | 1              |
| `vitimas_suicidios`                       | float   | Suicídios                                                | 3              |
| `vitimas_totais`                          | float   | Soma de todas as categorias de vítimas                   | 18             |

---

## **7. Pré-Processamento**

### ✔ **Tratamento de Chaves**

* `codigo_municipio_dv` → convertido para `str` em todas as tabelas.
* Padronização do código de Brasília, agregando regiões administrativas.

### ✔ **Agregação da População**

* Dados do Censo 2022 somados por município.
* Mantida população constante para todos os anos (hipótese explícita).

### ✔ **Pivot das Ocorrências Criminais**

Transformação de dados longos (`evento`, `total_vitimas`) para formato wide, criando colunas:

* `vitimas_feminicidio`
* `vitimas_homicidio_doloso`
* ...
* `vitimas_suicidios`
* `vitimas_totais`

### ✔ **Tratamento de Ausências**

* Eventos ausentes geram colunas preenchidas com 0.

### ✔ **Cálculo de Taxas (no script exploratório)**

Criadas colunas:

```
{crime}_por100mil = vítimas / Total_Habitantes * 100000
```

### ✔ **Integração dos Dados**

1. Merge entre pivot e PIB via:

   * `codigo_municipio_dv_agrupado`
   * `ano == ano_pib`
2. Merge com população via:

   * `codigo_municipio_dv_agrupado`

### ✔ **Filtros e Ordenação**

* Seleção estrita das 27 colunas finais.
* Ordenação por ano e município.

---

## **8. Estrutura Final do Dataset**

* **233 linhas** (33 municípios × 7 anos, exceto casos com dados faltantes originais)
* **27 colunas**
* Observações normalizadas e padronizadas

---

## **9. Observações Importantes**

### **Limitações:**

* População constante por município através dos anos → taxa por 100 mil pode distorcer tendência temporal.
* Dados de PIB normalmente têm defasagem temporal; merge assume equivalência direta.
* Eventos criminais raros (como feminicídio e latrocínio) geram baixa variabilidade estatística.

### **Riscos para Modelagem:**

* Multicolinearidade forte entre variáveis econômicas (`vl_servicos`, `vl_bruto_total`, `vl_pib`).
* Amostra pequena (233 linhas).
* Muitos zeros em crimes raros → modelagem deve usar técnicas adequadas (ex.: Poisson, regressão penalizada).

---

## **10. Resultado: Dataset Consolidado Pronto para Modelagem**

O dataset final integra:

* Indicadores econômicos
* Indicadores populacionais
* Indicadores criminais
* Taxas derivadas
* Estrutura temporal de 2015 a 2021

Permitindo análises como:

* correlações
* regressão linear e log-linear
* random forest
* análises comparativas por UF
* estudo longitudinal
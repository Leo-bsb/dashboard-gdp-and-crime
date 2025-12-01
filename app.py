# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import joblib
import os
from data_ingestion import load_data
from data_processing import prepare_data_for_modeling


st.write("CWD:", os.getcwd())
st.write("Arquivos no CWD:", os.listdir())
st.write("Arquivos no diretório do script:", os.listdir(os.path.dirname(__file__)))


'''# Configuração da página
st.set_page_config(
    page_title="Análise de Criminalidade - RIDE/DF",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Carregar dados
@st.cache_data
def load_cached_data():
    return load_data()

# Carregar modelo
@st.cache_resource
def load_model():
    model_path = 'models/best_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

df = load_cached_data()
model_data = load_model()

# Sidebar para navegação
st.sidebar.title("📑 Navegação")
page = st.sidebar.radio(
    "Selecione uma página:",
    ["🏠 Introdução", "📊 Análise Exploratória", "🤖 Modelagem Preditiva", "🎯 Fazer Predição"]
)

# ============================================================================
# PÁGINA 1: INTRODUÇÃO E CONTEXTUALIZAÇÃO
# ============================================================================
if page == "🏠 Introdução":
    st.markdown('<p class="main-header">📊 Análise de Criminalidade - RIDE/DF</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Problema de Pesquisa
    
    **Questão Central:** É possível reduzir a criminalidade de uma comunidade a partir de investimentos 
    em desenvolvimento econômico? Existe relação entre indicadores econômicos municipais e taxas de criminalidade?
    
    ### 📋 Contexto do Projeto
    
    Este projeto investiga a relação entre desenvolvimento econômico e criminalidade nos municípios da 
    Região Integrada de Desenvolvimento do Distrito Federal e Entorno (RIDE/DF).
    
    ### 📚 Bases de Dados Utilizadas
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h4>🚨 Ocorrências</h4>
        <p>Dados de criminalidade do SINESP via DataIESB</p>
        <ul>
            <li>Homicídios</li>
            <li>Feminicídios</li>
            <li>Latrocínios</li>
            <li>Outros crimes</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>💰 PIB Municipal</h4>
        <p>Indicadores econômicos do DataIESB</p>
        <ul>
            <li>PIB total</li>
            <li>PIB per capita</li>
            <li>Setores econômicos</li>
            <li>Valor adicionado</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h4>👥 Censo 2022</h4>
        <p>Dados populacionais do IBGE via DataIESB</p>
        <ul>
            <li>População total</li>
            <li>Densidade demográfica</li>
            <li>Distribuição por município</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 🔬 Metodologia
    
    1. **Coleta de Dados:** Integração de três bases de dados (Ocorrências, PIB e Censo)
    2. **Análise Exploratória:** Identificação de padrões e correlações
    3. **Feature Engineering:** Cálculo de taxas por 100 mil habitantes para normalização
    4. **Modelagem Preditiva:** Desenvolvimento de modelos de Machine Learning para prever criminalidade
    5. **Avaliação:** Comparação de modelos usando R², RMSE e validação cruzada
    
    ### 📈 Principais Descobertas
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **💡 Insight 1: População é o Principal Preditor**
        
        O total de habitantes apresenta forte correlação com criminalidade absoluta (R² > 0.80), 
        indicando que municípios maiores tendem a ter mais crimes em números absolutos.
        """)
    
    with col2:
        st.warning("""
        **📉 Insight 2: PIB Per Capita tem Baixa Correlação**
        
        O PIB per capita apresenta correlação muito fraca com taxas de criminalidade, 
        sugerindo que riqueza média individual não é um bom preditor de segurança pública.
        """)
    
    # Estatísticas gerais
    st.markdown('<p class="section-header">📊 Visão Geral dos Dados</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Registros", f"{len(df):,}")
    
    with col2:
        st.metric("Municípios", df['municipio_agrupado'].nunique())
    
    with col3:
        st.metric("Período", f"{df['ano'].min()} - {df['ano'].max()}")
    
    with col4:
        st.metric("Total de Vítimas", f"{df['vitimas_totais'].sum():,}")

# ============================================================================
# PÁGINA 2: ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
# ============================================================================
elif page == "📊 Análise Exploratória":
    st.markdown('<p class="main-header">📊 Análise Exploratória de Dados</p>', unsafe_allow_html=True)
    
    # Calcular taxas por 100 mil habitantes
    cols_crimes = ['vitimas_feminicidio', 'vitimas_homicidio_doloso',
                   'vitimas_tentativa_homicidio', 'vitimas_totais',
                   'vitimas_lesao_corporal_seguida_de_morte',
                   'vitimas_transito_ou_decorrencia_dele', 'vitimas_sem_indicio_de_crime',
                   'vitimas_latrocinio', 'vitimas_suicidios']
    
    for col in cols_crimes:
        df[f'{col}_por100mil'] = df[col] / df['Total_Habitantes'] * 100000
    
    # Seção 1: Visualização dos Dados
    st.markdown('<p class="section-header">🔍 Explorar Dataset</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Filtros")
        selected_uf = st.multiselect(
            "Selecione UF(s):",
            options=sorted(df['uf'].unique()),
            default=None
        )
        
        year_range = st.slider(
            "Período:",
            min_value=int(df['ano'].min()),
            max_value=int(df['ano'].max()),
            value=(int(df['ano'].min()), int(df['ano'].max()))
        )
    
    # Aplicar filtros
    df_filtered = df.copy()
    if selected_uf:
        df_filtered = df_filtered[df_filtered['uf'].isin(selected_uf)]
    df_filtered = df_filtered[(df_filtered['ano'] >= year_range[0]) & 
                              (df_filtered['ano'] <= year_range[1])]
    
    with col2:
        st.markdown(f"### Dados Filtrados ({len(df_filtered)} registros)")
        st.dataframe(df_filtered.head(10), use_container_width=True, height=300)
    
    # Estatísticas Descritivas
    st.markdown('<p class="section-header">📈 Estatísticas Descritivas</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Variáveis Numéricas")
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        selected_var = st.selectbox("Selecione uma variável:", numeric_cols)
        
        desc_stats = df_filtered[selected_var].describe()
        st.dataframe(desc_stats.to_frame(), use_container_width=True)
    
    with col2:
        st.markdown("### Distribuição")
        fig_hist = px.histogram(
            df_filtered, 
            x=selected_var,
            title=f"Distribuição de {selected_var}",
            color_discrete_sequence=['#1f77b4']
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Gráficos Interativos
    st.markdown('<p class="section-header">📊 Visualizações Interativas</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Por UF", "🔥 Correlações", "📉 Taxas Padronizadas", "📅 Evolução Temporal"])
    
    with tab1:
        st.markdown("### Total de Crimes por UF")
        df_uf = df_filtered.groupby('uf', as_index=False)['vitimas_totais'].sum().sort_values('vitimas_totais', ascending=False)
        
        fig1 = px.bar(
            df_uf, 
            x='uf', 
            y='vitimas_totais',
            title='Total de Crimes por Unidade Federativa',
            labels={'vitimas_totais': 'Total de Vítimas', 'uf': 'UF'},
            color='vitimas_totais',
            color_continuous_scale='Reds'
        )
        fig1.update_layout(height=500)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.info("💡 **Observação:** A criminalidade absoluta é maior em municípios mais populosos. " 
                "Para comparações justas entre municípios, use taxas padronizadas por 100 mil habitantes.")
    
    with tab2:
        st.markdown("### Matriz de Correlação")
        
        corr_option = st.radio(
            "Escolha o tipo de correlação:",
            ["Variáveis Econômicas e Criminalidade", "PIB per capita × Taxas de Crime"]
        )
        
        if corr_option == "Variáveis Econômicas e Criminalidade":
            cols_selecionadas = [
                'vl_agropecuaria', 'vl_industria', 'vl_servicos', 'vl_administracao',
                'vl_bruto_total', 'vl_subsidios', 'vl_pib', 'vl_pib_per_capta',
                'Total_Habitantes'
            ] + cols_crimes
            
            numeric_df = df_filtered[cols_selecionadas]
            corr = numeric_df.corr().round(2)
            
            fig_corr = ff.create_annotated_heatmap(
                z=corr.values,
                x=list(corr.columns),
                y=list(corr.index),
                colorscale='Greens',
                zmin=-1, zmax=1,
                showscale=True
            )
            fig_corr.update_layout(
                title="Correlação: Variáveis Econômicas × Criminalidade", 
                width=1200, height=800
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            st.markdown("""
            **📌 Interpretação:**
            - PIB total apresenta alta correlação com criminalidade absoluta (municípios maiores)
            - PIB per capita mostra baixa correlação, indicando que riqueza média não prediz criminalidade
            - População é o fator mais correlacionado com crimes totais
            """)
        
        else:
            colunas_taxas = [f'{c}_por100mil' for c in cols_crimes] + ['vl_pib_per_capta']
            corr_taxas = df_filtered[colunas_taxas].corr()
            
            fig_taxas = go.Figure(data=go.Heatmap(
                z=corr_taxas.values,
                x=corr_taxas.columns,
                y=corr_taxas.columns,
                colorscale='Blues',
                zmin=-1,
                zmax=1,
                showscale=True,
                text=corr_taxas.values.round(2),
                texttemplate="%{text}"
            ))
            fig_taxas.update_layout(
                title='Correlação: PIB per capita × Taxas de Crime (por 100 mil hab.)', 
                width=900, height=700
            )
            st.plotly_chart(fig_taxas, use_container_width=True)
            
            st.markdown("""
            **📌 Interpretação:**
            - Quando normalizamos por população, a correlação do PIB per capita com criminalidade permanece fraca
            - Isso sugere que desenvolvimento econômico individual não é suficiente para reduzir criminalidade
            - Outros fatores sociais e estruturais podem ser mais relevantes
            """)
    
    with tab3:
        st.markdown("### Taxas de Vítimas por 100 mil Habitantes")
        
        colunas_taxas_vitimas = [f'{c}_por100mil' for c in cols_crimes]
        df_media = df_filtered[colunas_taxas_vitimas].mean().sort_values()
        
        fig_vitimas = px.bar(
            df_media, 
            x=df_media.values, 
            y=df_media.index, 
            orientation='h',
            title='Taxas Médias de Vítimas por 100 mil habitantes',
            labels={'x': 'Taxa por 100 mil habitantes', 'y': 'Tipo de Crime'},
            color=df_media.values,
            color_continuous_scale='Oranges'
        )
        fig_vitimas.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_vitimas, use_container_width=True)
        
        st.success("✅ **Vantagem da Padronização:** Permite comparar municípios independentemente do tamanho populacional.")
    
    with tab4:
        st.markdown("### Evolução Temporal da Criminalidade")
        
        crime_type = st.selectbox(
            "Selecione o tipo de crime:",
            options=cols_crimes,
            format_func=lambda x: x.replace('vitimas_', '').replace('_', ' ').title()
        )
        
        df_temporal = df_filtered.groupby('ano')[crime_type].sum().reset_index()
        
        fig_temporal = px.line(
            df_temporal,
            x='ano',
            y=crime_type,
            title=f'Evolução de {crime_type.replace("vitimas_", "").replace("_", " ").title()} ao Longo do Tempo',
            markers=True
        )
        fig_temporal.update_layout(height=400)
        st.plotly_chart(fig_temporal, use_container_width=True)

# ============================================================================
# PÁGINA 3: MODELAGEM PREDITIVA
# ============================================================================
elif page == "🤖 Modelagem Preditiva":
    st.markdown('<p class="main-header">🤖 Modelagem Preditiva e Comparação de Modelos</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Objetivo da Modelagem
    
    Verificar se variáveis como **população** e **indicadores econômicos** podem prever a 
    **criminalidade total** de um município. Foram testados três algoritmos de regressão.
    """)
    
    # Resultados dos Modelos
    if model_data:
        st.markdown('<p class="section-header">📊 Comparação de Modelos</p>', unsafe_allow_html=True)
        
        results_df = pd.DataFrame(model_data['all_results']).T
        results_df = results_df.round(4)
        
        # Formatar para exibição
        display_df = results_df[['r2_test', 'rmse_test', 'mae_test', 'cv_r2_mean', 'cv_r2_std']].copy()
        display_df.columns = ['R² (Teste)', 'RMSE (Teste)', 'MAE (Teste)', 'CV R² (Média)', 'CV R² (Std)']
        
        # Destacar melhor modelo
        best_idx = display_df['R² (Teste)'].idxmax()
        
        st.dataframe(
            display_df.style.highlight_max(subset=['R² (Teste)'], color='lightgreen')
                           .highlight_min(subset=['RMSE (Teste)', 'MAE (Teste)'], color='lightblue'),
            use_container_width=True
        )
        
        st.success(f"🏆 **Melhor Modelo:** {model_data['model_name']} com R² = {model_data['metrics']['r2_test']:.4f}")
        
        # Gráfico de Comparação
        col1, col2 = st.columns(2)
        
        with col1:
            fig_r2 = px.bar(
                x=display_df.index,
                y=display_df['R² (Teste)'],
                title='Comparação de R² entre Modelos',
                labels={'x': 'Modelo', 'y': 'R² (Teste)'},
                color=display_df['R² (Teste)'],
                color_continuous_scale='Viridis'
            )
            fig_r2.update_layout(showlegend=False)
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col2:
            fig_rmse = px.bar(
                x=display_df.index,
                y=display_df['RMSE (Teste)'],
                title='Comparação de RMSE entre Modelos',
                labels={'x': 'Modelo', 'y': 'RMSE (Teste)'},
                color=display_df['RMSE (Teste)'],
                color_continuous_scale='Reds_r'
            )
            fig_rmse.update_layout(showlegend=False)
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        # Interpretação
        st.markdown('<p class="section-header">📝 Interpretação dos Resultados</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
            <h4>📈 Regressão Linear</h4>
            <p><strong>R² ≈ 0.81</strong></p>
            <p>Modelo simples que captura bem a relação linear entre população e criminalidade. 
            Explica ~81% da variação nos dados.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
            <h4>🌲 Random Forest</h4>
            <p><strong>R² ≈ 0.75</strong></p>
            <p>Modelo mais complexo, mas não superou a regressão linear. Isso indica que a 
            relação é predominantemente linear.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
            <h4>🚀 Gradient Boosting</h4>
            <p><strong>Desempenho intermediário</strong></p>
            <p>Modelo ensemble que tenta capturar não-linearidades, mas confirma que a 
            relação principal é linear.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.info("""
        **💡 Conclusão:** A regressão linear foi o melhor modelo, indicando que:
        1. A **população** é o preditor dominante de criminalidade absoluta
        2. A relação é **linear** e não requer modelos complexos
        3. Variáveis econômicas têm **impacto limitado** quando controlamos por população
        """)
        
    else:
        st.warning("⚠️ Modelo não encontrado. Execute o script `modeling.py` primeiro.")

# ============================================================================
# PÁGINA 4: FAZER PREDIÇÃO
# ============================================================================
elif page == "🎯 Fazer Predição":
    st.markdown('<p class="main-header">🎯 Fazer Predição Interativa</p>', unsafe_allow_html=True)
    
    if model_data:
        st.markdown(f"""
        Utilize o modelo **{model_data['model_name']}** (R² = {model_data['metrics']['r2_test']:.4f}) 
        para prever o número de vítimas em um município hipotético.
        """)
        
        st.markdown('<p class="section-header">📝 Insira os Dados do Município</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            total_habitantes = st.number_input(
                "👥 Total de Habitantes",
                min_value=1000,
                max_value=3000000,
                value=50000,
                step=1000,
                help="População total do município"
            )
            
            vl_pib_per_capta = st.number_input(
                "💰 PIB per Capita (R$)",
                min_value=5000.0,
                max_value=150000.0,
                value=25000.0,
                step=1000.0,
                help="PIB per capita em reais"
            )
            
            vl_agropecuaria = st.number_input(
                "🌾 Valor Agropecuária (R$ mil)",
                min_value=0.0,
                max_value=10000000.0,
                value=50000.0,
                step=10000.0,
                help="Valor adicionado do setor agropecuário"
            )
        
        with col2:
            vl_industria = st.number_input(
                "🏭 Valor Indústria (R$ mil)",
                min_value=0.0,
                max_value=50000000.0,
                value=100000.0,
                step=10000.0,
                help="Valor adicionado do setor industrial"
            )
            
            vl_servicos = st.number_input(
                "🏢 Valor Serviços (R$ mil)",
                min_value=0.0,
                max_value=100000000.0,
                value=300000.0,
                step=10000.0,
                help="Valor adicionado do setor de serviços"
            )
        
        # Botão de predição
        if st.button("🔮 Fazer Predição", type="primary", use_container_width=True):
            # Preparar dados
            input_data = pd.DataFrame({
                'Total_Habitantes': [total_habitantes],
                'vl_pib_per_capta': [vl_pib_per_capta],
                'vl_agropecuaria': [vl_agropecuaria],
                'vl_industria': [vl_industria],
                'vl_servicos': [vl_servicos]
            })
            
            # Fazer predição
            prediction = model_data['model'].predict(input_data)[0]
            
            # Exibir resultado
            st.markdown('<p class="section-header">📊 Resultado da Predição</p>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🎯 Vítimas Previstas", f"{int(prediction):,}")
            
            with col2:
                taxa = (prediction / total_habitantes) * 100000
                st.metric("📈 Taxa por 100 mil hab.", f"{taxa:.2f}")
            
            with col3:
                # Classificação
                if taxa < 50:
                    nivel = "🟢 Baixo"
                elif taxa < 100:
                    nivel = "🟡 Médio"
                else:
                    nivel = "🔴 Alto"
                st.metric("⚠️ Nível de Risco", nivel)
            
            # Comparação com média
            media_geral = df['vitimas_totais'].mean()
            diff_percent = ((prediction - media_geral) / media_geral) * 100
            
            if diff_percent > 0:
                st.warning(f"⚠️ Este município teria **{diff_percent:.1f}% mais vítimas** que a média geral ({media_geral:.0f} vítimas).")
            else:
                st.success(f"✅ Este município teria **{abs(diff_percent):.1f}% menos vítimas** que a média geral ({media_geral:.0f} vítimas).")
            
            # Gráfico de comparação
            fig_comp = go.Figure()
            
            fig_comp.add_trace(go.Bar(
                x=['Média Geral', 'Predição'],
                y=[media_geral, prediction],
                marker_color=['lightblue', 'darkblue'],
                text=[f'{media_geral:.0f}', f'{prediction:.0f}'],
                textposition='auto',
            ))
            
            fig_comp.update_layout(
                title='Comparação com Média Geral',
                yaxis_title='Número de Vítimas',
                height=400
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)
        
        # Análise de Sensibilidade
        st.markdown('<p class="section-header">🔬 Análise de Sensibilidade</p>', unsafe_allow_html=True)
        
        st.markdown("Veja como a predição muda ao variar uma variável, mantendo as outras constantes:")
        
        var_sensibilidade = st.selectbox(
            "Selecione a variável para análise:",
            ['Total_Habitantes', 'vl_pib_per_capta', 'vl_industria', 'vl_servicos']
        )
        
        # Criar range de valores
        base_values = {
            'Total_Habitantes': total_habitantes,
            'vl_pib_per_capta': vl_pib_per_capta,
            'vl_agropecuaria': vl_agropecuaria,
            'vl_industria': vl_industria,
            'vl_servicos': vl_servicos
        }
        
        if var_sensibilidade == 'Total_Habitantes':
            var_range = np.linspace(10000, 500000, 50)
        elif var_sensibilidade == 'vl_pib_per_capta':
            var_range = np.linspace(10000, 100000, 50)
        else:
            var_range = np.linspace(0, base_values[var_sensibilidade] * 3, 50)
        
        predictions_sensitivity = []
        for val in var_range:
            temp_values = base_values.copy()
            temp_values[var_sensibilidade] = val
            temp_df = pd.DataFrame([temp_values])
            pred = model_data['model'].predict(temp_df)[0]
            predictions_sensitivity.append(pred)
        
        fig_sens = px.line(
            x=var_range,
            y=predictions_sensitivity,
            title=f'Impacto de {var_sensibilidade} na Predição',
            labels={'x': var_sensibilidade, 'y': 'Vítimas Previstas'}
        )
        fig_sens.update_traces(line_color='#1f77b4', line_width=3)
        st.plotly_chart(fig_sens, use_container_width=True)
        
    else:
        st.error("❌ Modelo não encontrado. Por favor, execute o script `modeling.py` para treinar os modelos primeiro.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📖 Sobre
Este dashboard foi desenvolvido como projeto acadêmico para análise da relação entre 
desenvolvimento econômico e criminalidade na RIDE/DF.

**Dados:** DataIESB (SINESP, PIB Municipal, Censo 2022)

**Tecnologias:** Python, Streamlit, Scikit-learn, Plotly
""")'''

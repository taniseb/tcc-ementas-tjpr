import streamlit as st
import pandas as pd
import sqlite3
import base64
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="TCC - Resultados Ementas TJPR", layout="wide")

# Function to load data from SQLite
@st.cache_data
def load_data():
    conn = sqlite3.connect('ementas.db')
    df = pd.read_sql_query("SELECT * FROM ementas", conn)
    conn.close()
    return df

# Function to map classificacao_final_agrupada to simplified labels
def map_classificacao_final_agrupada(df):
    mapping = {
        '3. Manutenção da Sentença': 'Mantida',
        '4. Modificação da Sentença': 'Reformada',
        '5. Fora do Escopo': 'Fora do Escopo',
        '1. Fora do Escopo': 'Fora do Escopo',
        '6. Não Conhecida': 'Não Conhecida',
        '2. Hipóteses de não conhecimento da remessa': 'Não Conhecida'
    }
    df['classificacao_final'] = df['classificacao_final_agrupada'].map(mapping).fillna('Outros')
    return df

# Function to aggregate pre_classificacao
def aggregate_pre_classificacao(df):
    df['pre_classificacao_agrupada'] = df['pre_classificacao'].str.lower()
    df['pre_classificacao_agrupada'] = df['pre_classificacao_agrupada'].apply(
        lambda x: 'Mantida (Pré)' if 'mantida' in x else
                  'Reformada (Pré)' if 'reformada' in x else
                  'Fora do Escopo (Pré)' if 'fora do escopo' in x else
                  'Não Conhecida (Pré)' if 'não conhecida' in x else 'Outros (Pré)'
    )
    return df

# Function to map somente_remessa to 1/0, handling all data types
def map_somente_remessa(df):
    # Debug: Print unique values before mapping
    print("Unique values in somente_remessa before mapping:", df['somente_remessa'].unique().tolist())
    # Convert all possible values to string for consistent mapping
    df['somente_remessa_display'] = df['somente_remessa'].astype(str).map({
        '1': '1', '0': '0',
        'True': '1', 'False': '0',
        'Sim': '1', 'Não': '0',
        'true': '1', 'false': '0',
        'sim': '1', 'não': '0'
    }).fillna('Desconhecida')
    return df

# Function to map is_decisao_monocratica to 1/0, handling all data types
def map_decisao_monocratica(df):
    # Debug: Print unique values before mapping
    print("Unique values in is_decisao_monocratica before mapping:", df['is_decisao_monocratica'].unique().tolist())
    # Convert all possible values to string for consistent mapping
    df['decisao_monocratica_display'] = df['is_decisao_monocratica'].astype(str).map({
        '1': '1', '0': '0',
        'True': '1', 'False': '0',
        'true': '1', 'false': '0'
    }).fillna('Desconhecida')
    return df

# Function to rename area_tematica to area and handle NaN only for Mantida/Reformada
def prepare_area_column(df):
    if 'area_tematica' in df.columns:
        df = df.rename(columns={'area_tematica': 'area'})
        # Only keep area for Mantida and Reformada, set others to nan
        df['area'] = df.apply(
            lambda row: row['area'] if row['classificacao_final'] in ['Mantida', 'Reformada'] else pd.NA,
            axis=1
        )
    else:
        df['area'] = pd.NA
    return df

# Function to create downloadable link
def get_download_link(file_path, label):
    if not os.path.exists(file_path):
        return f"Arquivo {file_path} não encontrado."
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(file_path)}">{label}</a>'
    return href

# Function to render Chart.js chart using st.components.v1.html
def render_chart(chart_config, chart_id, height=400):
    # Convert numpy types to Python types recursively
    def convert_numpy(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        return obj

    # Apply conversion to the entire chart_config
    chart_config = convert_numpy(chart_config)
    chart_json = json.dumps(chart_config)
    html_code = f"""
    <div>
        <canvas id="{chart_id}" style="max-height: {height}px; width: 100%;"></canvas>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
    <script>
        const ctx_{chart_id} = document.getElementById('{chart_id}').getContext('2d');
        new Chart(ctx_{chart_id}, {chart_json});
    </script>
    """
    st.components.v1.html(html_code, height=height + 50)

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Sidebar navigation using buttons
st.sidebar.title("Navegação")

# Define the pages
pages = ["Home", "Resultados", "Análise Interativa", "Notebooks"]

# Create a button for each page
for page_name in pages:
    if st.sidebar.button(page_name, key=page_name):
        st.session_state.page = page_name

# Set the current page based on session state
page = st.session_state.page

# Load data
df = load_data()

# Drop precisa_revisao and revisado columns
df = df.drop(columns=['precisa_revisao', 'revisado'], errors='ignore')

# Map classificacao_final_agrupada, rename area_tematica, and map somente_remessa and is_decisao_monocratica
df = map_classificacao_final_agrupada(df)
df = prepare_area_column(df)
df = map_somente_remessa(df)
df = map_decisao_monocratica(df)

# Home Page
if page == "Home":
    st.title("TCC - Resultados Ementas TJPR")
    st.markdown("""
    Este site apresenta os resultados do meu TCC, que analisou 4.483 decisões judiciais do TJPR em 2024,
    focando na remessa necessária. Explore os resultados, interaja com os dados e acesse os notebooks para mais detalhes.
    """)
    st.markdown(get_download_link('tcc.pdf', 'VERSÃO A SER APROVADA PELA BANCA, EM ANÁLISE'), unsafe_allow_html=True)
    st.markdown(get_download_link('tcc_ap.pdf', 'VERSÃO DA APRESENTAÇÃO'), unsafe_allow_html=True)

# Results Page
elif page == "Resultados":
    st.title("Resultados do TCC")
    # Aggregate pre_classificacao for summary
    df = aggregate_pre_classificacao(df)
    pre_summary = df['pre_classificacao_agrupada'].value_counts()
    total_cases = len(df)
    pre_summary_percent = (pre_summary / total_cases * 100).round(2)
    
    # Final summary using actual counts
    final_summary = df['classificacao_final'].value_counts()
    final_summary_percent = (final_summary / total_cases * 100).round(2)
    
    # Build the summary text, excluding "Outros" if its count is 0
    st.markdown("### Resumo dos Resultados")
    st.markdown(f"""
    A análise de 4.483 casos do TJPR em 2024 revelou as seguintes classificações preliminares (agrupadas):  
    - **Mantida (Pré)**: {pre_summary_percent.get('Mantida (Pré)', 0)}% ({pre_summary.get('Mantida (Pré)', 0)} casos)  
    - **Reformada (Pré)**: {pre_summary_percent.get('Reformada (Pré)', 0)}% ({pre_summary.get('Reformada (Pré)', 0)} casos)  
    - **Fora do Escopo (Pré)**: {pre_summary_percent.get('Fora do Escopo (Pré)', 0)}% ({pre_summary.get('Fora do Escopo (Pré)', 0)} casos)  
    - **Não Conhecida (Pré)**: {pre_summary_percent.get('Não Conhecida (Pré)', 0)}% ({pre_summary.get('Não Conhecida (Pré)', 0)} casos)  
    - **Outros (Pré)**: {pre_summary_percent.get('Outros (Pré)', 0)}% ({pre_summary.get('Outros (Pré)', 0)} casos)  

    Após revisão, as classificações finais foram:  
    - **Mantida**: {final_summary_percent.get('Mantida', 0)}% ({final_summary.get('Mantida', 0)} casos)  
    - **Reformada**: {final_summary_percent.get('Reformada', 0)}% ({final_summary.get('Reformada', 0)} casos)  
    - **Fora do Escopo**: {final_summary_percent.get('Fora do Escopo', 0)}% ({final_summary.get('Fora do Escopo', 0)} casos)  
    - **Não Conhecida**: {final_summary_percent.get('Não Conhecida', 0)}% ({final_summary.get('Não Conhecida', 0)} casos)  
    """)
    # Only display "Outros" if its count is greater than 0
    if final_summary.get('Outros', 0) > 0:
        st.markdown(f"- **Outros**: {final_summary_percent.get('Outros', 0)}% ({final_summary.get('Outros', 0)} casos)  ")
    st.markdown("As reformas foram mais frequentes em áreas como direito tributário, conforme gráficos abaixo.")
    
    # Link to TCC PDF
    st.markdown(get_download_link('tcc.pdf', 'VERSÃO A SER APROVADA PELA BANCA, EM ANÁLISE'), unsafe_allow_html=True)
    
    # Chart 1: Agrupamentos dos registros da Remessa Necessária (Pie Chart)
    st.markdown("### Agrupamentos dos Registros da Remessa Necessária")
    counts = df['classificacao_final'].value_counts()
    fig = {
        'type': 'pie',
        'data': {
            'labels': counts.index.tolist(),
            'datasets': [{
                'data': counts.tolist(),
                'backgroundColor': ['#36a2eb', '#ff6384', '#ffce56', '#4bc0c0', '#9966ff'],
                'borderColor': ['#ffffff'],
                'borderWidth': 1
            }]
        },
        'options': {
            'responsive': True,
            'plugins': {
                'legend': {'position': 'top'},
                'title': {'display': True, 'text': 'Agrupamentos dos Registros da Remessa Necessária'}
            }
        }
    }
    render_chart(fig, "agrupamentos_chart")
    
    # Chart 2: Hipóteses de Não Cabimento - Somente Remessa e Decisão Monocrática
    st.markdown("### Hipóteses de Não Cabimento - Somente Remessa e Decisão Monocrática")
    nao_conhecida_df = df[df['classificacao_final'] == 'Não Conhecida']
    if not nao_conhecida_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Debug: Print the counts to understand the data
            remessa_counts = nao_conhecida_df['somente_remessa_display'].value_counts()
            st.write("Contagem de 'Somente Remessa' para 'Não Conhecida':", remessa_counts.to_dict())
            sim_count = remessa_counts.get('1', 0)
            nao_count = remessa_counts.get('0', 0)
            fig_remessa = {
                'type': 'bar',
                'data': {
                    'labels': ['1', '0'],
                    'datasets': [
                        {
                            'label': '1',
                            'data': [sim_count, 0],
                            'backgroundColor': '#36a2eb',
                            'borderColor': '#ffffff',
                            'borderWidth': 1
                        },
                        {
                            'label': '0',
                            'data': [0, nao_count],
                            'backgroundColor': '#ff6384',
                            'borderColor': '#ffffff',
                            'borderWidth': 1
                        }
                    ]
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'legend': {'position': 'top'},
                        'title': {'display': True, 'text': 'Somente Remessa'}
                    },
                    'scales': {
                        'y': {'beginAtZero': True, 'title': {'display': True, 'text': 'Contagem'}},
                        'x': {'title': {'display': True, 'text': 'Somente Remessa'}}
                    }
                }
            }
            render_chart(fig_remessa, "nao_conhecida_remessa_chart")
        
        with col2:
            # Debug: Print the counts to understand the data
            monocratica_counts = nao_conhecida_df['decisao_monocratica_display'].value_counts()
            st.write("Contagem de 'Decisão Monocrática' para 'Não Conhecida':", monocratica_counts.to_dict())
            sim_count = monocratica_counts.get('1', 0)
            nao_count = monocratica_counts.get('0', 0)
            fig_monocratica = {
                'type': 'bar',
                'data': {
                    'labels': ['1', '0'],
                    'datasets': [
                        {
                            'label': '1',
                            'data': [sim_count, 0],
                            'backgroundColor': '#36a2eb',
                            'borderColor': '#ffffff',
                            'borderWidth': 1
                        },
                        {
                            'label': '0',
                            'data': [0, nao_count],
                            'backgroundColor': '#ff6384',
                            'borderColor': '#ffffff',
                            'borderWidth': 1
                        }
                    ]
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'legend': {'position': 'top'},
                        'title': {'display': True, 'text': 'Decisão Monocrática'}
                    },
                    'scales': {
                        'y': {'beginAtZero': True, 'title': {'display': True, 'text': 'Contagem'}},
                        'x': {'title': {'display': True, 'text': 'Decisão Monocrática'}}
                    }
                }
            }
            render_chart(fig_monocratica, "nao_conhecida_monocratica_chart")
    else:
        st.write("Nenhum dado disponível para 'Não Conhecida'.")

    # Chart 3: Subclassificações de Hipóteses de Não Cabimento (Pie Chart using Matplotlib)
    st.markdown("### Subclassificações de Hipóteses de Não Cabimento")
    if not nao_conhecida_df.empty and 'subclassificacao_nao_conhecimento' in df.columns:
        # Calculate the distribution from the base
        subclass_counts = nao_conhecida_df['subclassificacao_nao_conhecimento'].value_counts()
        
        # Print the distribution for confirmation
        st.write("Distribuição das Subclassificações (calculada a partir da base):")
        for category, count in subclass_counts.items():
            st.write(f"{count} casos para {category}")
        st.write(f"Total de casos 'Não Conhecimento': {len(nao_conhecida_df)}")
        
        # Prepare data for the chart
        labels = subclass_counts.index.tolist()
        values = subclass_counts.values.tolist()

        # Define colors for each subcategory (distinctive for light/dark themes)
        colors = [
            "#1f77b4",  # Blue
            "#ff7f0e",  # Orange
            "#2ca02c",  # Green
            "#d62728",  # Red
            "#9467bd",  # Purple
            "#8c564b",  # Brown
            "#e377c2",  # Pink
            "#7f7f7f",  # Gray
        ]

        # Ensure colors match the number of categories
        colors = colors * (len(labels) // len(colors) + 1)
        colors = colors[:len(labels)]

        # Create a pie chart
        plt.figure(figsize=(10, 8))
        plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 10})
        plt.title("Distribuição das Subclassificações de Não Conhecimento (224 Casos)", fontsize=12, pad=20)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

        # Save the chart
        plt.savefig("subclassificacao_pie_chart.png", dpi=300, bbox_inches='tight')

        # Display the chart in Streamlit
        st.image("subclassificacao_pie_chart.png", caption="Gráfico de Pizza das Subclassificações de Não Conhecimento", use_container_width=True)
    else:
        st.write("Nenhum dado disponível para subclassificação de 'Não Conhecida'.")

    # Chart 4: Manutenção da Sentença - Somente Remessa e Decisão Monocrática
    st.markdown("### Manutenção da Sentença - Somente Remessa e Decisão Monocrática")
    mantida_df = df[df['classificacao_final'] == 'Mantida']
    if not mantida_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Debug: Print the counts to understand the data
            remessa_counts = mantida_df['somente_remessa_display'].value_counts()
            st.write("Contagem de 'Somente Remessa' para 'Mantida':", remessa_counts.to_dict())
            sim_count = remessa_counts.get('1', 0)
            nao_count = remessa_counts.get('0', 0)
            fig_remessa = {
                'type': 'bar',
                'data': {
                    'labels': ['1', '0'],
                    'datasets': [
                        {
                            'label': '1',
                            'data': [sim_count, 0],
                            'backgroundColor': '#36a2eb',
                            'borderColor': '#ffffff',
                            'borderWidth': 1
                        },
                        {
                            'label': '0',
                            'data': [0, nao_count],
                            'backgroundColor': '#ff6384',
                            'borderColor': '#ffffff',
                            'borderWidth': 1
                        }
                    ]
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'legend': {'position': 'top'},
                        'title': {'display': True, 'text': 'Somente Remessa'}
                    },
                    'scales': {
                        'y': {'beginAtZero': True, 'title': {'display': True, 'text': 'Contagem'}},
                        'x': {'title': {'display': True, 'text': 'Somente Remessa'}}
                    }
                }
            }
            render_chart(fig_remessa, "mantida_remessa_chart")
        
        with col2:
            # Debug: Print the counts to understand the data
            monocratica_counts = mantida_df['decisao_monocratica_display'].value_counts()
            st.write("Contagem de 'Decisão Monocrática' para 'Mantida':", monocratica_counts.to_dict())
            sim_count = monocratica_counts.get('1', 0)
            nao_count = monocratica_counts.get('0', 0)
            fig_monocratica = {
                'type': 'bar',
                'data': {
                    'labels': ['1', '0'],
                    'datasets': [
                        {
                            'label': '1',
                            'data': [sim_count, 0],
                            'backgroundColor': '#36a2eb',
                            'borderColor': '#ffffff',
                            'borderWidth': 1
                        },
                        {
                            'label': '0',
                            'data': [0, nao_count],
                            'backgroundColor': '#ff6384',
                            'borderColor': '#ffffff',
                            'borderWidth': 1
                        }
                    ]
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'legend': {'position': 'top'},
                        'title': {'display': True, 'text': 'Decisão Monocrática'}
                    },
                    'scales': {
                        'y': {'beginAtZero': True, 'title': {'display': True, 'text': 'Contagem'}},
                        'x': {'title': {'display': True, 'text': 'Decisão Monocrática'}}
                    }
                }
            }
            render_chart(fig_monocratica, "mantida_monocratica_chart")
    else:
        st.write("Nenhum dado disponível para 'Mantida'.")

    # Chart 5: Modificação da Sentença - Somente Remessa e Decisão Monocrática
    st.markdown("### Modificação da Sentença - Somente Remessa e Decisão Monocrática")
    reformada_df = df[df['classificacao_final'] == 'Reformada']
    if not reformada_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Debug: Print the counts to understand the data
            remessa_counts = reformada_df['somente_remessa_display'].value_counts()
            st.write("Contagem de 'Somente Remessa' para 'Reformada':", remessa_counts.to_dict())
            sim_count = remessa_counts.get('1', 0)
            nao_count = remessa_counts.get('0', 0)
            fig_remessa = {
                'type': 'bar',
                'data': {
                    'labels': ['1', '0'],
                    'datasets': [
                        {
                            'label': '1',
                            'data': [sim_count, 0],
                            'backgroundColor': '#36a2eb',
                            'borderColor': '#ffffff',
                            'borderWidth': 1
                        },
                        {
                            'label': '0',
                            'data': [0, nao_count],
                            'backgroundColor': '#ff6384',
                            'borderColor': '#ffffff',
                            'borderWidth': 1
                        }
                    ]
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'legend': {'position': 'top'},
                        'title': {'display': True, 'text': 'Somente Remessa'}
                    },
                    'scales': {
                        'y': {'beginAtZero': True, 'title': {'display': True, 'text': 'Contagem'}},
                        'x': {'title': {'display': True, 'text': 'Somente Remessa'}}
                    }
                }
            }
            render_chart(fig_remessa, "reformada_remessa_chart")
        
        with col2:
            # Debug: Print the counts to understand the data
            monocratica_counts = reformada_df['decisao_monocratica_display'].value_counts()
            st.write("Contagem de 'Decisão Monocrática' para 'Reformada':", monocratica_counts.to_dict())
            sim_count = monocratica_counts.get('1', 0)
            nao_count = monocratica_counts.get('0', 0)
            fig_monocratica = {
                'type': 'bar',
                'data': {
                    'labels': ['1', '0'],
                    'datasets': [
                        {
                            'label': '1',
                            'data': [sim_count, 0],
                            'backgroundColor': '#36a2eb',
                            'borderColor': '#ffffff',
                            'borderWidth': 1
                        },
                        {
                            'label': '0',
                            'data': [0, nao_count],
                            'backgroundColor': '#ff6384',
                            'borderColor': '#ffffff',
                            'borderWidth': 1
                        }
                    ]
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'legend': {'position': 'top'},
                        'title': {'display': True, 'text': 'Decisão Monocrática'}
                    },
                    'scales': {
                        'y': {'beginAtZero': True, 'title': {'display': True, 'text': 'Contagem'}},
                        'x': {'title': {'display': True, 'text': 'Decisão Monocrática'}}
                    }
                }
            }
            render_chart(fig_monocratica, "reformada_monocratica_chart")
    else:
        st.write("Nenhum dado disponível para 'Reformada'.")

    # Chart 6: Sentenças Mantidas e Modificadas por Área (Bar Chart)
    st.markdown("### Sentenças Mantidas e Modificadas por Área")
    mantidas_reformadas = df[df['classificacao_final'].isin(['Mantida', 'Reformada'])]
    counts = mantidas_reformadas.groupby(['area', 'classificacao_final']).size().reset_index(name='Contagem')
    areas_list = counts['area'].unique().tolist()
    mantidas_full = [int(counts[(counts['area'] == area) & (counts['classificacao_final'] == 'Mantida')]['Contagem'].iloc[0]) if area in counts[counts['classificacao_final'] == 'Mantida']['area'].values else 0 for area in areas_list]
    reformadas_full = [int(counts[(counts['area'] == area) & (counts['classificacao_final'] == 'Reformada')]['Contagem'].iloc[0]) if area in counts[counts['classificacao_final'] == 'Reformada']['area'].values else 0 for area in areas_list]
    
    mantidas_reformadas_fig = {
        'type': 'bar',
        'data': {
            'labels': areas_list,
            'datasets': [
                {
                    'label': 'Mantida',
                    'data': mantidas_full,
                    'backgroundColor': '#36a2eb',
                    'borderColor': '#ffffff',
                    'borderWidth': 1
                },
                {
                    'label': 'Reformada',
                    'data': reformadas_full,
                    'backgroundColor': '#ff6384',
                    'borderColor': ['#ffffff'],
                    'borderWidth': 1
                }
            ]
        },
        'options': {
            'responsive': True,
            'plugins': {
                'legend': {'position': 'top'},
                'title': {'display': True, 'text': 'Sentenças Mantidas e Modificadas por Área'}
            },
            'scales': {
                'y': {'beginAtZero': True, 'title': {'display': True, 'text': 'Contagem'}},
                'x': {'title': {'display': True, 'text': 'Área'}}
            }
        }
    }
    render_chart(mantidas_reformadas_fig, "mantidas_reformadas_chart")
    
    # Chart 7: Distribuição por Áreas (Pie Chart for Each Classification)
    st.markdown("### Distribuição por Áreas (por Classificação)")
    classifications = ['Mantida', 'Reformada']
    for classification in classifications:
        subset_df = df[df['classificacao_final'] == classification]
        if not subset_df.empty:
            area_counts = subset_df['area'].value_counts()
            fig = {
                'type': 'pie',
                'data': {
                    'labels': area_counts.index.tolist(),
                    'datasets': [{
                        'data': area_counts.tolist(),
                        'backgroundColor': ['#36a2eb', '#ff6384', '#ffce56', '#4bc0c0', '#9966ff', '#ff9f40'],
                        'borderColor': ['#ffffff'],
                        'borderWidth': 1
                    }]
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'legend': {'position': 'top'},
                        'title': {'display': True, 'text': f'Áreas - {classification}'}
                    }
                }
            }
            render_chart(fig, f"area_pie_chart_{classification.lower().replace(' ', '_')}")
        else:
            st.write(f"Nenhum dado disponível para {classification}.")

    # Chart 8: Divisão por Áreas (Bar Chart, only Mantida/Reformada)
    st.markdown("### Divisão por Áreas (Sentenças Mantidas e Modificadas)")
    area_df = df[df['classificacao_final'].isin(['Mantida', 'Reformada'])]
    if not area_df.empty:
        area_counts = area_df['area'].value_counts()
        area_data = {
            'Área': area_counts.index.tolist(),
            'Contagem': area_counts.tolist()
        }
        area_fig = {
            'type': 'bar',
            'data': {
                'labels': area_data['Área'],
                'datasets': [{
                    'label': 'Contagem',
                    'data': area_data['Contagem'],
                    'backgroundColor': ['#36a2eb', '#ff6384', '#ffce56', '#4bc0c0', '#9966ff', '#ff9f40'],
                    'borderColor': ['#ffffff'],
                    'borderWidth': 1
                }]
            },
            'options': {
                'responsive': True,
                'plugins': {
                    'legend': {'position': 'top'},
                    'title': {'display': True, 'text': 'Distribuição por Áreas'}
                },
                'scales': {
                    'y': {'beginAtZero': True, 'title': {'display': True, 'text': 'Contagem'}},
                    'x': {'title': {'display': True, 'text': 'Área'}}
                }
            }
        }
        render_chart(area_fig, "area_chart")
    else:
        st.write("Nenhum dado disponível para Sentenças Mantidas e Modificadas.")

# Interactive Analysis Page
elif page == "Análise Interativa":
    # Scroll to the top when the page is loaded
    st.components.v1.html(
        """
        <script>
            window.scrollTo(0, 0);
        </script>
        """,
        height=0
    )
    
    st.title("Análise Interativa")
    st.markdown("Filtre os dados e crie visualizações personalizadas. Aqui, você pode ver a classificação feita por mim, ler a ementa, modificá-la, como preferir! Divirta-se! ")
    
    # Filters
    # Use area_tematica for the filter
    classificacao = st.multiselect("Classificação Final:", options=df['classificacao_final'].unique(), default=[])
    area = st.multiselect("Área:", options=df['area_tematica'].dropna().unique() if 'area_tematica' in df.columns else [], default=[])
    somente_remessa = st.multiselect("Somente Remessa:", options=['1', '0'], default=[])
    # Add filter for subclassificacao_nao_conhecimento
    subclassificacao = st.multiselect(
        "Subclassificação de Não Cabimento:",
        options=df['subclassificacao_nao_conhecimento'].dropna().unique(),
        default=[]
    )
    
    # Apply filters
    filtered_df = df
    if classificacao:
        filtered_df = filtered_df[filtered_df['classificacao_final'].isin(classificacao)]
    if area and 'area_tematica' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['area_tematica'].isin(area)]
    if somente_remessa:
        filtered_df = filtered_df[filtered_df['somente_remessa_display'].isin(somente_remessa)]
    if subclassificacao:
        filtered_df = filtered_df[filtered_df['subclassificacao_nao_conhecimento'].isin(subclassificacao)]
    
    # Display the total after filters
    st.markdown(f"### Total de casos filtrados: {len(filtered_df)}")
    
    # Display the data after filters
    st.markdown("### Dados Filtrados")
    # Include numero_processo, ementa, and url (as clickable link) along with other columns
    # Use area_tematica instead of area to avoid NA values for non-Mantida/Reformada cases
    display_columns = ['numero_processo', 'ementa', 'url', 'area_tematica', 'classificacao_final', 'subclassificacao_nao_conhecimento']
    # Filter columns that exist in the DataFrame
    display_columns = [col for col in display_columns if col in filtered_df.columns]
    
    # Create a copy of the DataFrame for display with URL as clickable link
    display_df = filtered_df[display_columns].head(100).copy()
    if 'url' in display_df.columns:
        display_df['url'] = display_df['url'].apply(
            lambda x: f'<a href="{x}" target="_blank">{x}</a>' if pd.notnull(x) and x else 'N/A'
        )
    # Display the DataFrame with clickable URLs
    st.markdown(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    # Visualizations
    if not filtered_df.empty:
        chart_type = st.selectbox("Tipo de Gráfico:", ["Barra", "Pizza"])
        
        if chart_type == "Barra":
            st.markdown("### Distribuição das Classificações por Área")
            # Use area_tematica for the chart as well
            if 'area_tematica' in filtered_df.columns:
                counts = filtered_df.groupby(['area_tematica', 'classificacao_final']).size().unstack(fill_value=0)
                areas = counts.index.tolist()
                classifications = counts.columns.tolist()
                datasets = [
                    {
                        'label': classification,
                        'data': counts[classification].tolist(),
                        'backgroundColor': ['#36a2eb', '#ff6384', '#ffce56', '#4bc0c0', '#9966ff', '#ff9f40'][i % 6],
                        'borderColor': '#ffffff',
                        'borderWidth': 1
                    }
                    for i, classification in enumerate(classifications)
                ]
                fig = {
                    'type': 'bar',
                    'data': {
                        'labels': areas,
                        'datasets': datasets
                    },
                    'options': {
                        'responsive': True,
                        'plugins': {
                            'legend': {'position': 'top'},
                            'title': {'display': True, 'text': 'Distribuição das Classificações por Área'}
                        },
                        'scales': {
                            'x': {'stacked': True, 'title': {'display': True, 'text': 'Área'}},
                            'y': {'stacked': True, 'beginAtZero': True, 'title': {'display': True, 'text': 'Contagem'}}
                        }
                    }
                }
                render_chart(fig, "interactive_bar_chart")
            else:
                st.write("Coluna 'area_tematica' não encontrada para gerar o gráfico.")
        
        elif chart_type == "Pizza":
            st.markdown("### Proporção das Classificações")
            counts = filtered_df['classificacao_final'].value_counts()
            fig = {
                'type': 'pie',
                'data': {
                    'labels': counts.index.tolist(),
                    'datasets': [{
                        'data': counts.tolist(),
                        'backgroundColor': ['#36a2eb', '#ff6384', '#ffce56', '#4bc0c0', '#9966ff'],
                        'borderColor': ['#ffffff'],
                        'borderWidth': 1
                    }]
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'legend': {'position': 'top'},
                        'title': {'display': True, 'text': 'Proporção das Classificações'}
                    }
                }
            }
            render_chart(fig, "interactive_pie_chart")

# Notebooks Page
elif page == "Notebooks":
    st.title("Notebooks com o Código Utilizado")
    st.markdown("""
    Para leitores interessados em explorar mais detalhes do código aplicado, disponibilizei dois notebooks Python:
    - **tcc_extracao.ipynb**: Contém o processo completo de extração e classificação inicial dos dados. Aqui, é para fazer o download dos dados e salvá-los.
    - **tcc_analise.ipynb**: Inclui a análise detalhada e geração dos gráficos apresentados. Neste passo, todo o processo levou em conta a leitura -manual- e análise dos processos.  
    
    Você também pode baixar a base de dados em formato SQLite para análise própria.
    """)
    st.markdown(get_download_link('tcc_extracao.ipynb', 'Baixar tcc_extracao.ipynb'), unsafe_allow_html=True)
    st.markdown(get_download_link('tcc_analise.ipynb', 'Baixar tcc_analise.ipynb'), unsafe_allow_html=True)
    st.markdown(get_download_link('ementas.db', 'Baixar Banco de Dados (SQLite)'), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Desenvolvido por Tanise Brandão Bussmann para o Trabalho de Conclusão de Curso de Direito - 2025")
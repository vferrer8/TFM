# F1-GPT — Asistente conversacional para análisis y predicción en Fórmula 1

Trabajo de Fin de Máster · Máster Universitario en Inteligencia Artificial · UTAMED  
Autor: **Víctor Ferrer Teixidor** · Director: **Alfons Marquès**

---

F1-GPT es un asistente conversacional especializado en Fórmula 1 que combina razonamiento basado en LLM con datos históricos estructurados, telemetría en tiempo real y predicción de posición de carrera mediante aprendizaje automático.

El sistema responde tanto en español como en inglés, adapta el tipo de respuesta según si la consulta es descriptiva, predictiva o mixta, y rechaza cualquier pregunta fuera del dominio de la F1.

## Arquitectura general

El agente orquesta cuatro capas:

```
┌──────────────┐      ┌──────────────────────────────────────────┐
│  Streamlit   │◀────▶│              F1Agent                     │
│ (frontend)   │      │  ┌────────────┐                          │
└──────────────┘      │  │DomainGuard │  ← filtro de dominio     │
                      │  └─────┬──────┘                          │
                      │        ▼                                  │
                      │  ┌────────────┐    ┌──────────────────┐  │
                      │  │  Gemini    │◀──▶│   Tool calling   │  │
                      │  │ Flash 2.0  │    │  · get_driver…   │  │
                      │  └─────┬──────┘    │  · get_race…     │  │
                      │        │           │  · get_telemetry │  │
                      │        │           │  · predict_pos…  │  │
                      │  ┌─────┴──────┐    └────────┬─────────┘  │
                      │  │ToolValidator│            │            │
                      │  └─────┬──────┘            │            │
                      │        ▼                    ▼            │
                      │  ┌──────────────┐  ┌─────────────────┐  │
                      │  │ Consistency  │  │ SQLite · OpenF1 │  │
                      │  │   Guard      │  │  XGBoost model  │  │
                      │  └──────────────┘  └─────────────────┘  │
                      └──────────────────────────────────────────┘
```

**Capa semántica** — El system prompt embebe el mapeo de entidades (pilotos, circuitos, constructores) y las reglas de selección de herramientas. El LLM nunca genera SQL ni llama a la API directamente; delega en las tools.

**Tool calling** — El SDK de Gemini gestiona el ciclo completo de invocación. Hay cuatro herramientas disponibles:

| Herramienta | Fuente de datos | Uso |
|---|---|---|
| `get_driver_standings` | SQLite (Ergast 1950–2024) | Ficha y datos básicos de un piloto |
| `get_race_results` | SQLite (Ergast 1950–2024) | Resultados oficiales de un GP |
| `get_telemetry` | OpenF1 API (2023+) | Tiempos de vuelta y telemetría reciente |
| `predict_position` | Modelo XGBoost | Estimación de posición final de carrera |

**Guardarraíles** — Tres filtros encadenados con interfaz uniforme `{"valid": bool, "message": str}`:

| Guardarraíl | Momento | Función |
|---|---|---|
| `DomainGuard` | Pre-inferencia | Filtra consultas fuera del dominio F1 mediante allowlist léxica bilingüe (ES/EN) |
| `ToolValidator` | Pre-ejecución | Verifica que cada tool invocada está en la allowlist; bloquea inyección de prompt |
| `ConsistencyGuard` | Post-generación | Detecta números en la respuesta que no aparecen en los datos devueltos por las tools |

## Modelo predictivo

El módulo de predicción emplea un modelo **XGBoost** entrenado sobre el histórico de Ergast (1950–2024). Predice la posición final de un piloto a partir de diez variables pre-carrera:

| Feature | Descripción |
|---|---|
| `grid` | Posición de salida |
| `year` | Temporada |
| `season_progress` | Ronda actual / total de rondas |
| `circuit_alt` | Altitud del circuito (m) |
| `driver_age` | Edad del piloto en el momento de la carrera |
| `driver_experience` | Número de GPs disputados hasta esa carrera |
| `avg_finish_last3` | Media de posiciones finales en las últimas 3 carreras |
| `driver_pos_before` | Posición en el campeonato antes de la carrera |
| `driver_wins_before` | Victorias acumuladas en la temporada |
| `constructor_pos_before` | Posición del constructor antes de la carrera |

Las variables que no pueden obtenerse directamente de la base de datos (carreras futuras o pilotos sin histórico) se imputan con la mediana del dataset. El resultado se recorta al intervalo [1, 25] y se acompaña siempre de un aviso de limitación del modelo.

## Evaluación

El banco de pruebas (`evaluation/benchmark.json`) contiene 20 consultas de referencia distribuidas en cuatro categorías:

- **descriptive** — consultas sobre resultados históricos y fichas de piloto
- **predictive** — estimaciones de posición futura
- **comparative** — cruces entre temporadas o pilotos
- **out_of_domain** — consultas ajenas a la F1, que deben ser rechazadas

El script `evaluate.py` mide task completion rate (global y por categoría), latencia extremo a extremo (media, p50, p90) y coste estimado por consulta.

## Estructura del repositorio

```
Project/
├── streamlit_app.py        # Frontend conversacional (entrypoint principal)
├── config.py               # Variables de entorno y rutas de modelos y BD
├── requirements.txt        # Dependencias fijadas
├── train_xgboost_tfm.py    # Entrenamiento del modelo XGBoost
├── evaluate.py             # Evaluación automática del agente
├── evaluation/
│   └── benchmark.json      # 20 consultas de referencia con respuesta esperada
├── data/
│   ├── f1_dataset_v2.csv   # Dataset preprocesado para entrenamiento
│   ├── f1_assistant.db     # Base de datos SQLite (dump Ergast)
│   ├── F1_XGBoost_TFM.ipynb         # Notebook de desarrollo del modelo
│   └── F1_Comparativa_Modelos.ipynb # Notebook comparativa LR / RF / XGBoost
├── models/
│   └── race_pos_model.pkl  # Modelo XGBoost serializado con sus encoders
└── src/
    ├── agent/
    │   └── f1_agent.py           # Orquestador: LLM + tool calling + guardarraíles
    ├── api/
    │   ├── db_client.py          # Cliente SQLite (datos históricos Ergast)
    │   └── f1_client.py          # Cliente OpenF1 (telemetría 2023+)
    ├── guards/
    │   ├── domain_guard.py       # Filtro léxico de dominio (ES/EN)
    │   ├── consistency_guard.py  # Verificación numérica post-generación
    │   └── tool_validator.py     # Allowlist de herramientas permitidas
    ├── ml/
    │   ├── predictor.py          # Wrapper de inferencia del modelo XGBoost
    │   └── feature_resolver.py   # Resolución de las 10 features desde SQLite
    └── tools/
        └── f1_tools.py           # Funciones expuestas al LLM via tool calling
```

## Fuentes de datos

- **Ergast Motor Racing Data** — dump histórico de resultados 1950–2024, almacenado en SQLite local (`data/f1_assistant.db`).
- **OpenF1 API** — telemetría y tiempos de vuelta de sesiones a partir de 2023. Se consulta en tiempo real únicamente cuando el usuario solicita datos de telemetría reciente.

## Limitaciones conocidas

- La capa semántica está embebida en el system prompt. Pilotos o circuitos no contemplados en los mapas de entidades pueden producir errores de resolución.
- El SDK de Gemini gestiona el ciclo de tool calling de forma autónoma, por lo que el `ConsistencyGuard` opera en modo *best-effort* sobre la respuesta final: los datos intermedios de las herramientas no quedan expuestos durante la generación.
- El cliente OpenF1 expone únicamente `get_laps`; otros endpoints (sectores, intervalos, condiciones meteorológicas) no están implementados.

## Licencia

Uso académico, sin fines comerciales. Datos históricos cortesía de los dumps públicos de Ergast y la API OpenF1.

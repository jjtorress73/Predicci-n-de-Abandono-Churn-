# Predicci-n-de-Abandono-Churn-
Modelo creado para la predicción de abandono de clientes
***Informe final*** — Predicción de abandono de clientes Interconnect (Churn)
1. ***Objetivo del proyecto***

El objetivo del proyecto fue desarrollar un modelo de Machine Learning capaz de identificar clientes con alta probabilidad de abandonar el servicio de ***Interconnect***, de manera que la empresa pueda anticiparse al abandono mediante promociones, planes especiales u otras estrategias de retención.

La solución se planteó como un problema de clasificación binaria, donde:

0 → cliente que permanece.
1 → cliente que abandona (churn).

La evaluación se enfocó principalmente en F1-score y ROC-AUC, considerando que en un problema de churn es importante lograr un equilibrio entre identificar correctamente a los clientes que abandonarán y evitar demasiados falsos positivos. Aunque podrian evaluarse mediante métrica ***Recall*** si los costos de abandono vs retención asi lo justifica.

2. ***Pasos del plan realizados y omitidos***

***Pasos realizados***

2.1. ***Exploración y comprensión de los datos***

Se realizó un análisis exploratorio **EDA** para:

* Conocer la estructura y dimensiones de los datos.
* Revisar tipos de variables.
* Identificar valores faltantes.
* Analizar variables numéricas y categóricas.
* Estudiar la variable objetivo churn.
* Analizar la relación entre variables y abandono.

Entre las variables analizadas estuvieron:
* MonthlyCharges   
* TotalCharges   
* tenure_days   
* Type   
* PaymentMethod   
* InternetService   
* OnlineSecurity   
* TechSupport   
* StreamingTV   
* StreamingMovies   
* Variables creadas y de agrupamiento  

También se analizaron correlaciones entre las variables numéricas. Se observó, por ejemplo, una correlación elevada entre TotalCharges y tenure_days, por lo que se consideró la posible redundancia entre ambas variables.

2.2. ***Preparación de los datos***

Se construyó un esquema de preprocesamiento mediante ColumnTransformer.

Para las variables numéricas se utilizó:
StandardScaler() para variables numéricas y OneHotEncoder() para categóricas

Esto permitió que los modelos recibieran correctamente tanto variables numéricas como categóricas.

El preprocesamiento se integró dentro de un Pipeline, evitando realizar transformaciones de manera independiente y reduciendo el riesgo de data leakage.

2.3. ***Separación de entrenamiento, validación y prueba***
Se mantuvo un conjunto de TEST separado para la evaluación final.

El procedimiento utilizado fue:
```
Datos
  │
  ├── Entrenamiento → selección de hiperparámetros
  │
  ├── Validación → selección del threshold
  │
  └── TEST → evaluación final
```
2.4. ***Entrenamiento y comparación de diferentes modelos***

Se evaluaron diferentes enfoques:

* Dummy
* Dummy Stratified
* Regresión Logística
* Regresión Logística balanceada
* Decision Tree
* Random Forest
* XGBoost
* LightGBM
* CatBoost

Esto permitió comparar modelos simples contra algoritmos de ensamble y boosting.

2.5. Optimización de hiperparámetros

Para los modelos principales se utilizó GridSearchCV con validación cruzada.

Esto permitió buscar sistemáticamente las combinaciones de hiperparámetros que proporcionaban un mejor desempeño.

Por ejemplo, para XGBoost se obtuvo:
```
{
    'model__colsample_bytree': 1.0,
    'model__learning_rate': 0.1,
    'model__max_depth': 7,
    'model__min_child_weight': 5,
    'model__n_estimators': 300,
    'model__subsample': 0.8
}
```
El mejor F1 obtenido durante la validación cruzada fue aproximadamente:
F1 CV = 0.7630
2.6. ***Optimización del threshold***

No se utilizó automáticamente el threshold tradicional de 0.5.

Se calcularon diferentes valores de threshold utilizando el conjunto de validación y se seleccionó aquel que maximizaba el F1-score.

Conceptualmente:
```
mejor_threshold = df_threshold.loc[
    df_threshold['f1'].idxmax()
]
```
Esto permitió adaptar la clasificación al comportamiento del problema de churn.

Posteriormente, el threshold seleccionado se mantuvo fijo para evaluar TEST.

2.7. ***Evaluación final***

Finalmente, los modelos se evaluaron sobre el conjunto TEST utilizando:

Accuracy
Precision
Recall
F1-score
ROC-AUC

El resultado permitió comparar el desempeño de todos los modelos bajo las mismas condiciones.
***Pasos omitidos***

No todos los análisis posibles de Machine Learning fueron necesarios para resolver el problema.

***KNN***

No se consideró necesario incorporar KNN como modelo final porque los modelos de boosting presentaron un desempeño claramente superior.

Agregar modelos adicionales sin una justificación de negocio o metodológica no necesariamente mejora la solución.

Optimización del threshold sobre TEST

Este paso no se realizó intencionalmente.

Optimizar el threshold utilizando TEST produciría una contaminación del conjunto de evaluación y podría generar una estimación demasiado optimista del desempeño real.

***PCA**

No se utilizó PCA como etapa principal del modelo final porque el preprocesamiento mediante One-Hot Encoding y los algoritmos de árboles/boosting permiten trabajar directamente con las variables disponibles.

Además, PCA dificultaría la interpretación de las variables originales.
3. ***Principales dificultades encontradas y solución***
3.1. ***Diferencias entre entornos y versiones***

Una de las dificultades más importantes fue la incompatibilidad al intentar cargar modelos previamente guardados mediante joblib.

Se detectaron diferencias entre:

Python
NumPy
Scikit-learn
XGBoost
Joblib

El problema provocaba errores al cargar los archivos .pkl.   

***Solución***

Se identificó que el notebook estaba utilizando un entorno .venv diferente al entorno donde se había entrenado el modelo.

Se verificaron:   
* sys.executable)
* numpy.__version__)
* sklearn.__version__)
* xgboost.__version__)
* joblib.__version__)

Esto permitió detectar que el kernel utilizado por Jupyter no correspondía al entorno donde se había guardado el modelo.

También se consideró como alternativa reconstruir el pipeline con los mismos hiperparámetros y volver a entrenarlo cuando el modelo original no pudiera recuperarse.
3.2. ***Variables categóricas en XGBoost***

Otra dificultad apareció al intentar entrenar XGBoost directamente sobre variables categóricas.

El modelo no podía recibir directamente valores de tipo object.

Solución

Se utilizó un Pipeline manual para que el modelo reciba finalmente los datos adecuados para su procesamiento.   

3.3. ***Selección del threshold***

Utilizar simplemente:0.50 no necesariamente proporciona el mejor equilibrio entre ***precision y recall***.

***Solución***    

Se construyó una tabla de thresholds y métricas y se seleccionó el valor que maximizó F1 en VALIDACIÓN.

Esto permitió que cada modelo tuviera su propio threshold óptimo.   

3.4. ***Diferencias entre modelos y métricas***

Se observó que un modelo puede tener un recall alto pero un precision menor.

Por ejemplo, Random Forest presentó un recall elevado, pero su F1 fue inferior al de los modelos de boosting.

Esto mostró que no se debe seleccionar el modelo únicamente por Accuracy o Recall.

Para este proyecto fue más adecuado considerar el equilibrio entre:

Precision ↔ Recall

mediante F1-score, además del poder discriminativo medido por ROC-AUC.

4. ***Pasos clave para resolver la tarea***

Los pasos más importantes fueron:

1. **Preparar correctamente los datos**

Separar las variables predictoras y la variable objetivo y realizar correctamente el tratamiento de variables numéricas y categóricas.

2. **Evitar data leakage**

El preprocesamiento se integró dentro del pipeline y el conjunto TEST se mantuvo separado.

3. **Comparar varios modelos**

No se asumió inicialmente que un algoritmo sería el mejor. Se compararon modelos lineales, árboles, Random Forest y diferentes algoritmos de boosting.

4. **Optimizar hiperparámetros**

Se utilizó GridSearchCV para encontrar configuraciones más adecuadas para los modelos principales.

5. **Optimizar el threshold**

El threshold se seleccionó mediante VALIDACIÓN buscando maximizar F1.

6. **Evaluar sobre TEST**

Una vez seleccionado el modelo y su threshold, se realizó la evaluación final sobre datos no utilizados durante el entrenamiento ni durante la selección del threshold.

7. **Considerar la utilidad empresarial**

El objetivo no es solamente obtener una métrica elevada, sino identificar clientes con riesgo de abandono para permitir que Interconnect pueda actuar antes de que ocurra el churn.

5. ***Comparación final de modelos***

De acuerdo con los resultados obtenidos, el ranking por F1-score, de mayor a menor, es:


| Posición | Modelo            | Threshold |   Accuracy |  Precision |     Recall |   F1-score |    ROC-AUC |
| -------: | ----------------- | --------: | ---------: | ---------: | ---------: | ---------: | ---------: |
|     🥇 1 | **XGBoost**       | **0.320** | **89.35%** | **79.94%** | **79.94%** | **79.94%** | **93.59%** |
|     🥈 2 | LightGBM          |     0.360 |     89.35% |     80.77% |     78.61% |     79.67% |     93.04% |
|     🥉 3 | CatBoost          |     0.410 |     87.72% |     79.30% |     72.73% |     75.87% |     92.37% |
|        4 | Decision Tree     |     0.478 |     83.32% |     69.36% |     66.58% |     67.94% |     85.00% |
|        5 | Random Forest     |     0.273 |     75.09% |     51.95% | **81.82%** |     63.55% |     85.95% |
|        6 | Logistic          |     0.336 |     77.15% |     55.51% |     70.05% |     61.94% |     83.17% |
|        7 | Logistic Balanced |     0.550 |     80.06% |     66.79% |     49.47% |     56.84% |     83.02% |
|        8 | Dummy Stratified  |     0.500 |     59.83% |     24.47% |     24.60% |     24.53% |     48.58% |
|        9 | Dummy             |     0.500 |     73.46% |      0.00% |      0.00% |      0.00% |     50.00% |

```
6. ¿Cuál es el modelo final?
🥇 XGBoost

El modelo final seleccionado es XGBoost, utilizando el threshold de 0.320, de acuerdo con la última tabla de resultados obtenida.

Su configuración optimizada fue:
```
XGBClassifier(
    colsample_bytree=1.0,
    learning_rate=0.1,
    max_depth=7,
    min_child_weight=5,
    n_estimators=300,
    subsample=0.8,
    random_state=12345,
    eval_metric='logloss'
)
```
***Calidad del modelo***

En TEST obtuvo:
* Accuracy: 89.35%
* Precision: 79.94%
* Recall: 79.94%
* F1-score: 79.94%
* ROC-AUC: 93.59%

El ***ROC-AUC de 93.59%*** indica una capacidad de discriminación alta entre clientes que abandonan y clientes que permanecen.

El F1 de aproximadamente 80% también muestra un buen equilibrio entre identificar clientes que efectivamente abandonarán y limitar los falsos positivos.

7. **Interpretación para el negocio**

El modelo final permite generar una probabilidad de abandono para cada cliente.

Por ejemplo:
```
Cliente
   ↓
Características actuales
   ↓
XGBoost
   ↓
Probabilidad de churn
   ↓
¿Probabilidad ≥ 0.320?
       ↓
      Sí
       ↓
Cliente de alto riesgo
       ↓
Acción de retención
```

Esto permite que Interconnect pueda concentrar sus recursos en los clientes que presentan mayor riesgo.

Sin embargo, 0.320 no debe considerarse necesariamente el threshold definitivo para producción. Fue seleccionado optimizando F1. Para una implementación empresarial sería recomendable incorporar el análisis de costo-beneficio del churn, porque el costo de perder un cliente probablemente sea diferente al costo de ofrecer una promoción a un cliente que finalmente no abandona.

Por ejemplo:

Si el costo de perder un cliente es mucho mayor que el costo de ofrecerle una promoción, podría ser conveniente reducir el threshold para aumentar el recall y detectar más clientes potencialmente perdidos.

8. **Conclusión final**

El proyecto permitió construir una solución completa de Machine Learning para anticipar el abandono de clientes de Interconnect.

Se realizó el proceso de exploración y preparación de datos, construcción del pipeline, comparación de modelos, optimización de hiperparámetros, selección del threshold mediante validación y evaluación final sobre TEST.

Los principales modelos de boosting mostraron un desempeño considerablemente superior a los modelos base. XGBoost obtuvo el mejor F1-score y ROC-AUC, por lo que se selecciona como modelo final.

Con un F1-score de 79.94% y ROC-AUC de 93.59%, el modelo presenta un nivel de calidad alto para una primera solución predictiva de churn.

La solución no termina con la predicción: su valor empresarial está en utilizar la probabilidad de abandono para priorizar acciones de retención, buscando que el costo de las promociones sea menor que el costo esperado de perder clientes.

**Recomendación final**

La siguiente etapa debería ser pasar de:

“¿Quién probablemente abandona?”

a:

“¿A quién conviene ofrecer una promoción y cuánto estamos dispuestos a invertir para evitar su abandono?”

Para ello, el modelo predictivo debe complementarse con el análisis costo-beneficio de las acciones de retención, que permitirá seleccionar el threshold óptimo desde una perspectiva económica y no únicamente estadística.

Para lo anterior se puede plantear un horizonte analisis costo-beneficio de un año, año y medio, dos años dado que el comportamiento se observa que existe un abandono mayor en los primeros 6 meses de estancia en el servicio tomando como costo de oportunidad el pago promedio de los clientes pr tipo de servicio adquirido vs los ingresos poteniales de su retencion asi como los costos asociados para promover su retencion y promocion aplicada para realizarlo.

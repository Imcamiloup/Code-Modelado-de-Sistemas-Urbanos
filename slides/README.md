# PITCH COMPLETO - PRESENTACIÓN TRABAJO DE GRADO (20 minutos)

## DIAPOSITIVA 1: Portada (30 seg)

**Pitch:**
"Buenos días/tardes. Mi nombre es Luis Camilo Gómez y presento mi trabajo de grado: 'Modelo Integrado de Sistemas Urbanos aplicado a Bogotá'. En los próximos 20 minutos mostraré cómo desarrollé un sistema de ecuaciones diferenciales para explicar y proyectar la dinámica poblacional, desigualdad y bienestar en nuestra ciudad."

---

## DIAPOSITIVA 2-4: Contexto (2 min)

**Pitch:**
"Las ciudades son sistemas complejos donde población, desigualdad, bienestar, territorio e infraestructura ecológica interactúan constantemente. Bogotá no es la excepción: tiene 8 millones de habitantes, un Gini de 0.52 y enfrenta presión sobre sus recursos naturales.

El problema clave es que estos fenómenos NO evolucionan de forma independiente. Hay retroalimentaciones: la desigualdad afecta el crecimiento poblacional, el bienestar influye en la desigualdad, y la población presiona el territorio.

Necesitamos herramientas matemáticas rigurosas para entender estas dinámicas y poder proyectar escenarios futuros."

---

## DIAPOSITIVA 5: Objetivos (1 min)

**Pitch:**
"Mi objetivo general fue desarrollar y calibrar un modelo matemático que capture estas interacciones. Específicamente:

1. Formular un sistema de EDOs basado en teoría de sistemas urbanos
2. Calibrar parámetros con datos reales 2012-2024
3. Validar el ajuste histórico
4. Analizar estabilidad mediante teoría de fase
5. Proyectar escenarios a 5 años"

---

## DIAPOSITIVA 6: Introducción Marco Teórico (1 min)

**Pitch:**
"El marco teórico combina tres pilares: modelado logístico de población (inspirado en Verhulst), teoría de sistemas complejos (retroalimentaciones no lineales), y economía urbana (efectos de desigualdad y bienestar). La innovación es acoplar estas tres dimensiones en un solo sistema."

---

## DIAPOSITIVA 7: Variables del Sistema (1 min)

**Pitch:**
"Trabajé con 5 variables clave:

- P: Población (habitantes)
- F: Desigualdad (Gini)
- W: Bienestar (índice compuesto)
- U: Huella urbana (hectáreas construidas)
- E: Estructura ecológica (hectáreas protegidas)

Las primeras tres son endógenas (evolucionan dinámicamente). U y E se tratan como exógenas suavizadas por datos históricos."

---

## DIAPOSITIVA 8: Sistema de EDOs (1.5 min)

**Pitch:**
"Este es el corazón del modelo. Tres ecuaciones acopladas:

Ecuación de Población: crecimiento logístico clásico (término rP(1-P/K)) más el efecto de la desigualdad (gamma_F × F). Si la desigualdad es alta, frena el crecimiento.

Ecuación de Desigualdad: relajación natural (alpha_F) más la influencia del bienestar (beta_F × W). Si mejora el bienestar, la desigualdad debería reducirse.

Ecuación de Bienestar: impacto poblacional, efecto de infraestructura (U+E), autorregulación y penalización por desigualdad (delta_W × F).

Son 10 parámetros libres que debemos estimar con datos reales."

---

## DIAPOSITIVA 9-10: Ecuaciones Individuales (Población/Territorio) (1 min)

**Pitch:**
"Veamos en detalle: La población sigue una curva en S hasta llegar a K (capacidad de carga). Gamma_F negativo significa que alta desigualdad expulsa población (migración, reducción de natalidad).

El territorio (U) crece polinomialmente según datos históricos, sin retroalimentación dinámica en esta versión simplificada."

---

## DIAPOSITIVA 11: Métodos de Ajuste (1.5 min)

**Pitch:**
"Para calibrar necesitaba dos cosas: datos confiables y una buena estrategia de optimización.

Datos: DANE y Secretaría de Planeación de Bogotá, 13 años (2012-2024). Normalicé todo con z-score para que las variables sean comparables.

Optimización híbrida: primero Differential Evolution (explora globalmente el espacio de parámetros, evita mínimos locales), luego L-BFGS-B (refina localmente). La función de pérdida pondera por varianza inversa y da peso doble a desigualdad, que es la más volátil.

Resultado: pérdida final de 2.43 tras 500 iteraciones."

---

## DIAPOSITIVA 12: Análisis de Fase (2 min)

**Pitch:**
"Una vez calibrado, analicé la estabilidad. Busqué puntos fijos (donde las derivadas son cero) y calculé el Jacobiano.

Encontré dos puntos: PF1 es una silla (inestable), PF2 es un atractor estable. Los eigenvalores de PF2 son todos negativos: -1.00, -1.00, -0.22. Esto garantiza convergencia asintótica.

Los diagramas de fase muestran que desde cualquier condición inicial, el sistema fluye hacia PF2. No hay ciclos límite, no hay caos. Es un sistema predecible a largo plazo."

---

## DIAPOSITIVA 13: Proyecciones (1.5 min)

**Pitch:**
"Proyecté 5 años (2024-2029) bajo el escenario conservador: U y E constantes.

Resultados:

- Población converge a ~8M (saturación cerca de K)
- Desigualdad se relaja lentamente de 0.52 a 0.51
- Bienestar se estabiliza ligeramente por debajo de la media histórica

Esto sugiere que sin intervenciones en capacidad de carga o redistribución, Bogotá alcanzará un equilibrio subóptimo."

---

## DIAPOSITIVA 14: Referencias (30 seg - opcional)

**Pitch:**
"El trabajo se fundamenta en literatura clásica: Verhulst para logística, Forrester para sistemas urbanos, y estudios recientes sobre desigualdad urbana en América Latina."

---

## DIAPOSITIVAS 15-18: Estructura Ecológica, Bienestar, Implementación (2 min)

**Pitch:**
"Sobre E: suavicé con polinomio grado 3 porque tiene variaciones no lineales históricas (expansión y posterior estancamiento de áreas protegidas).

Bienestar W: índice compuesto normalizado. El parámetro delta_W=1.0 indica que la desigualdad lo penaliza fuertemente.

Implementación: Python con scipy (solve_ivp, fsolve), numpy, matplotlib. Todo el código es reproducible y está documentado."

---

## DIAPOSITIVA 19-20: Gráficos Validación (1 min)

**Pitch:**
"Aquí vemos el ajuste histórico. R² de población es 0.96 (excelente). Desigualdad y bienestar tienen R² más bajos (0.17 y 0.27) porque son inherentemente más volátiles, pero las tendencias están capturadas."

---

## DIAPOSITIVA 21: Datos (30 seg)

**Pitch:**
"Todos los datos son públicos y verificables: DANE para población y Gini, SDP Bogotá para huella urbana y estructura ecológica. Total: 13 puntos temporales anuales."

---

## DIAPOSITIVA 22: Gráfico Proyección (30 seg)

**Pitch:**
"Este gráfico resume las proyecciones. Noten cómo P se aplana (saturación), F baja suavemente y W se mantiene estable. Sin shocks externos, este es el equilibrio esperado."

---

## DIAPOSITIVA 23: Resultados Cuantitativos (1 min)

**Pitch:**
"Resultados clave:

- r=0.5 (crecimiento poblacional moderado)
- K=1.56 en z-score (~8M habitantes reales)
- gamma_F=-1.69 (desigualdad REDUCE población significativamente)
- delta_W=1.0 (desigualdad PENALIZA bienestar 1 a 1)
- beta_F=0.01 (bienestar apenas afecta desigualdad, aquí hay una desconexión importante)"

---

## DIAPOSITIVA 24: Limitaciones (1 min)

**Pitch:**
"Reconozco tres limitaciones principales:

1. U y E son exógenas (versión futura: 5 EDOs completas)
2. No captura shocks externos (pandemias, crisis económicas)
3. Ajuste débil en F y W (R² bajos) sugiere necesidad de términos no lineales (F², interacciones cruzadas)

Aun así, el modelo es útil para escenarios de largo plazo."

---

## DIAPOSITIVA 25: Recomendaciones de Política (1 min)

**Pitch:**
"Basado en los parámetros estimados, tres recomendaciones:

1. **Expandir K**: invertir en infraestructura (vivienda, servicios) para evitar saturación
2. **Fortalecer beta_F**: diseñar políticas que vinculen bienestar con reducción de desigualdad (actualmente es 0.01, muy débil)
3. **Reducir delta_W**: implementar redes de protección que amortigüen el impacto de desigualdad en bienestar"

---

## DIAPOSITIVA 26: Discusión (1 min)

**Pitch:**
"El modelo confirma que Bogotá tiene un atractor estable, lo cual es positivo (no hay colapso). Pero el equilibrio está cerca de capacidad de carga y con bienestar subóptimo.

La desigualdad actúa como freno poblacional Y como penalizadora de bienestar. Es la variable más crítica del sistema.

Comparado con modelos previos (Forrester, System Dynamics), este aporta rigor matemático en estabilidad y proyecciones cuantitativas."

---

## DIAPOSITIVAS 27-29: Conclusiones Finales (2 min)

**Pitch:**
"Para concluir:

**Logros:**

- Sistema calibrado y validado con R² poblacional de 0.96
- Estabilidad matemáticamente demostrada (eigenvalores negativos)
- Proyecciones coherentes con tendencias observadas
- Identificación de palancas de política (K, beta_F, delta_W)

**Contribución:**

- Primer modelo acoplado P-F-W para Bogotá con validación cuantitativa
- Marco replicable para otras ciudades latinoamericanas
- Herramienta de soporte a planeación urbana

**Trabajo futuro:**

- Incorporar U, E endógenas (modelo completo 5 EDOs)
- Análisis de bifurcaciones sobre parámetros críticos
- Bootstrap para intervalos de confianza
- Validación en Medellín, Cali, otras capitales

Gracias por su atención. Quedo atento a sus preguntas."

---

## TIMING TOTAL: ~20 minutos

- Introducción/Contexto: 4 min
- Modelo/Metodología: 5 min
- Resultados/Análisis: 6 min
- Discusión/Conclusiones: 5 min

---

## TIPS DE PRESENTACIÓN:

1. **Portada → Contexto**: Enganchar con el problema (retroalimentaciones urbanas)
2. **EDOs**: Pausar para que vean las ecuaciones, señalar términos clave
3. **Diagramas de fase**: Usar puntero láser en PF2 (atractor)
4. **Proyecciones**: Enfatizar saturación poblacional
5. **Conclusiones**: Reforzar palancas de política (aplicabilidad práctica)
6. **Dejar 3-5 min para preguntas**

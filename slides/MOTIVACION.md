# PITCH AMPLIADO: CONTEXTO Y MOTIVACIÓN (5-6 minutos)

## DIAPOSITIVAS 2-4: Contexto Ampliado

**[DIAPOSITIVA 2 - Inicio del contexto]**

**Pitch (Parte 1 - El Desafío Urbano Global):**

"Permítanme comenzar con una pregunta fundamental: ¿Por qué necesitamos modelos matemáticos para entender nuestras ciudades?

La respuesta está en la complejidad sin precedentes que enfrentamos. América Latina es la región más urbanizada del mundo en desarrollo: 8 de cada 10 personas viven en ciudades. Y Bogotá, con sus 8 millones de habitantes, es un laboratorio viviente de esta transformación urbana acelerada.

Pero aquí está el problema crítico: nuestras ciudades NO son la suma de sus partes. No podemos entender el crecimiento poblacional sin considerar la desigualdad. No podemos diseñar políticas de bienestar sin entender cómo la expansión urbana presiona la estructura ecológica. No podemos proyectar el futuro mirando variables aisladas.

**[Hacer pausa, dejar que la idea penetre]**

Las ciudades son SISTEMAS COMPLEJOS con retroalimentaciones no lineales. Y esta complejidad nos ha llevado históricamente a dos extremos problemáticos:

Por un lado, modelos conceptuales ricos pero sin poder predictivo cuantitativo. Por otro, análisis estadísticos que capturan correlaciones pero ignoran las causas estructurales.

**[DIAPOSITIVA 3 - Problema específico de Bogotá]**

**Pitch (Parte 2 - La Situación Crítica de Bogotá):**

Veamos los números de Bogotá que nos motivaron:

**Población:** De 6.8 millones en 2012 a 8 millones en 2024. Un crecimiento del 17% en 12 años. ¿Hasta dónde puede crecer? ¿Cuál es la capacidad de carga real?

**Desigualdad:** Gini de 0.52, uno de los más altos de América Latina. Y aquí está el hallazgo inquietante de estudios previos: la alta desigualdad NO solo es un problema social, es un FRENO al crecimiento poblacional. La gente migra, las familias reducen su tamaño, la ciudad expulsa población cuando la brecha se hace insostenible.

**Bienestar:** Índices estancados cerca de 99 sobre 100 (aparentemente alto), pero con volatilidad creciente. ¿Qué significa un índice alto si la desigualdad lo está erosionando constantemente?

**Territorio:** La huella urbana creció de 38,000 a 45,000 hectáreas. Mientras tanto, la estructura ecológica protegida se mantiene en ~15,000 hectáreas, una presión ambiental evidente.

**[Transición crítica - bajar el tono, volverse más serio]**

Estos números NO evolucionan independientemente. Déjenme mostrarles las retroalimentaciones que descubrimos:

**[DIAPOSITIVA 4 - Retroalimentaciones del sistema]**

**Pitch (Parte 3 - Las Retroalimentaciones Ocultas):**

**Primera retroalimentación - Desigualdad frena Población:**
Cuando la desigualdad aumenta (F↑), la población tiende a estancarse o decrecer (P↓). No es obvio, pero los datos lo confirman: en años de Gini alto, las tasas de crecimiento poblacional caen. ¿Por qué? Migración hacia ciudades intermedias, reducción de natalidad en hogares vulnerables, expulsión económica.

Este es el parámetro gamma_F que estimaremos: resultó ser -1.69. Negativo y fuerte. La desigualdad es un FRENO poblacional significativo.

**Segunda retroalimentación - Bienestar influye en Desigualdad (pero débilmente):**
Teoría económica dice que mejorar el bienestar promedio debería reducir la desigualdad (mediante políticas redistributivas implícitas). Pero encontramos que beta_F = 0.01. Casi cero.

Esto es un hallazgo CRÍTICO: Bogotá tiene una desconexión entre bienestar agregado y reducción de desigualdad. Mejoramos índices promedio, pero la brecha permanece.

**Tercera retroalimentación - Desigualdad penaliza Bienestar:**
Esta es la más fuerte: delta_W = 1.0. Por cada unidad que aumenta la desigualdad, el bienestar cae proporcionalmente. Es una relación 1 a 1, casi determinista.

Implicación: NO podemos mejorar el bienestar sostenidamente sin atacar la desigualdad. Son vasos comunicantes.

**Cuarta retroalimentación - Población presiona Bienestar:**
alpha_W = -0.01. Ligeramente negativo. A mayor población, leve caída en bienestar (congestión, presión sobre servicios). No es dramático, pero existe.

**Quinta retroalimentación - Infraestructura afecta Bienestar:**
beta_W = -0.08. Contraintuitivo: la suma de huella urbana y estructura ecológica (U+E) tiene efecto NEGATIVO en bienestar. Interpretación: posible multicolinealidad o captura de efectos de congestión y fragmentación ecológica.

**[Pausa estratégica - hacer contacto visual con el jurado]**

**Pitch (Parte 4 - Por Qué Este Modelo Es Necesario AHORA):**

Entonces, ¿por qué este proyecto es urgente y relevante?

**Razón 1 - Vacío Metodológico:**
No existe en la literatura un modelo matemático CALIBRADO con datos reales que integre población, desigualdad y bienestar para ciudades latinoamericanas. Los modelos de System Dynamics de Forrester (1969) son cualitativos. Los modelos econométricos urbanos capturan correlaciones pero no dinámicas.

Este trabajo llena ese vacío: EDOs calibradas, validación cuantitativa (R² de 0.96 en población), análisis de estabilidad matemática (teorema de Hartman-Grobman).

**Razón 2 - Herramienta de Política Pública:**
Bogotá gasta ~50 billones de pesos anuales en presupuesto. Las decisiones sobre vivienda, infraestructura, programas sociales se hacen con proyecciones lineales o escenarios ad-hoc.

Este modelo permite:

- Proyectar escenarios "qué pasaría si expandimos K (capacidad) en 10%"
- Evaluar "qué pasaría si logramos aumentar beta_F (vínculo bienestar-desigualdad) de 0.01 a 0.05"
- Identificar puntos de inflexión ANTES de que ocurran

Es una herramienta de soporte a decisión basada en evidencia, no en intuición.

**Razón 3 - Replicabilidad Regional:**
El marco es generalizable. Si funciona en Bogotá, puede calibrarse para Medellín, Cali, Ciudad de México, Lima, Buenos Aires.

Imaginemos una red de modelos urbanos acoplados en América Latina, compartiendo metodología, comparando parámetros. Podríamos identificar patrones regionales: ¿Todas las capitales tienen gamma_F negativo? ¿Beta_F es universalmente bajo en la región?

**Razón 4 - Fundamento Teórico Riguroso:**
No es un modelo "caja negra" de machine learning. Cada ecuación tiene justificación física:

- Ecuación P: biología poblacional (Verhulst 1838) + efectos socioeconómicos
- Ecuación F: relajación estructural + influencia de bienestar (economía del desarrollo)
- Ecuación W: función de producción urbana + externalidades

Los parámetros son interpretables. No solo predecimos, ENTENDEMOS.

**[DIAPOSITIVA 4 final - Transición a objetivos]**

**Pitch (Parte 5 - El Puente hacia la Solución):**

Por todo esto, el objetivo de este trabajo NO es solo académico. Es construir una herramienta que:

1. **Explique** por qué Bogotá llegó a donde está (validación histórica con R² > 0.9 en población)
2. **Prediga** hacia dónde va bajo escenarios plausibles (proyecciones a 5 años)
3. **Prescriba** qué palancas mover para cambiar el rumbo (análisis de sensibilidad paramétrico)

En los próximos 15 minutos les mostraré:

- Cómo formulé matemáticamente estas retroalimentaciones
- Cómo calibré 10 parámetros con 13 años de datos
- Qué descubrí sobre la estabilidad del sistema (hay un atractor único, predecible)
- Qué escenarios proyectamos para 2024-2029
- Y qué recomendaciones concretas emergen para política pública

Pero antes de entrar en ecuaciones, déjenme mostrarles el objetivo general y los específicos...

**[TRANSICIÓN NATURAL A DIAPOSITIVA 5 - Objetivos]**

---

## ELEMENTOS RETÓRICOS CLAVE EN ESTE PITCH AMPLIADO:

### 1. **Gancho Emocional-Racional:**

- Comienza con pregunta (involucra a la audiencia)
- Datos impactantes (8 de cada 10, 8 millones de habitantes)
- Contraste (conceptual vs. cuantitativo, correlación vs. causalidad)

### 2. **Estructura de Problema-Solución:**

- Problema global → Problema regional → Problema específico Bogotá
- Retroalimentaciones como "descubrimientos" (genera curiosidad)
- Vacío metodológico (justifica la contribución)

### 3. **Lenguaje Visual:**

- "Laboratorio viviente"
- "Vasos comunicantes"
- "Freno poblacional"
- "Desconexión entre bienestar y desigualdad"

### 4. **Anticipación de Preguntas:**

- "¿Por qué un modelo matemático?" → Porque las políticas actuales son ad-hoc
- "¿Por qué Bogotá?" → Porque es representativa y tiene datos
- "¿Para qué sirve?" → Soporte a decisión, replicable

### 5. **Transiciones Estratégicas:**

- Pausas después de datos críticos
- "Aquí está el hallazgo inquietante..."
- "Déjenme mostrarles..."
- "Por todo esto..."

---

## TIMING SUGERIDO PARA ESTAS 3 DIAPOSITIVAS:

- **Diapositiva 2 (Desafío urbano):** 1:30 min
- **Diapositiva 3 (Situación Bogotá):** 1:30 min
- **Diapositiva 4 (Retroalimentaciones + Por qué es necesario):** 2:30 min
- **Transición a objetivos:** 0:30 min

**Total:** 6 minutos (30% del tiempo total, justificado por la importancia del contexto)

---

## AJUSTE DEL TIMING TOTAL DE LA PRESENTACIÓN:

Con este contexto ampliado:

- **Contexto/Motivación:** 6 min (ampliado)
- **Objetivos:** 1 min
- **Marco teórico/Variables:** 1.5 min (reducido ligeramente)
- **Sistema EDOs:** 1.5 min
- **Metodología/Calibración:** 2 min
- **Resultados (parámetros + validación):** 2 min
- **Análisis de fase/Estabilidad:** 2 min
- **Proyecciones:** 1.5 min
- **Discusión/Limitaciones:** 1.5 min
- **Conclusiones/Recomendaciones:** 2 min

**Total:** 21 minutos + margen para preguntas

---

## DOCUMENTO DE APOYO VISUAL (si usas pizarra o puntero):

Durante el pitch de retroalimentaciones, puedes dibujar rápidamente:

```
    ↗ P (población)
   /     ↘
  /       ↘ (alpha_W: -0.01)
 /         ↘
F ← (beta_F: 0.01) ← W (bienestar)
 \         ↑
  \       / (delta_W: -1.0)
   ↘    /
    ↘ /
   (gamma_F: -1.69)
```

Y señalar: "Miren este ciclo: F penaliza a P y a W, pero W apenas afecta a F. Es un sistema desequilibrado hacia la desigualdad."

---

¿Quieres que desarrolle algún otro pitch específico con este nivel de detalle, o ajuste el timing de otras secciones?# PITCH AMPLIADO: CONTEXTO Y MOTIVACIÓN (5-6 minutos)

## DIAPOSITIVAS 2-4: Contexto Ampliado

**[DIAPOSITIVA 2 - Inicio del contexto]**

**Pitch (Parte 1 - El Desafío Urbano Global):**

"Permítanme comenzar con una pregunta fundamental: ¿Por qué necesitamos modelos matemáticos para entender nuestras ciudades?

La respuesta está en la complejidad sin precedentes que enfrentamos. América Latina es la región más urbanizada del mundo en desarrollo: 8 de cada 10 personas viven en ciudades. Y Bogotá, con sus 8 millones de habitantes, es un laboratorio viviente de esta transformación urbana acelerada.

Pero aquí está el problema crítico: nuestras ciudades NO son la suma de sus partes. No podemos entender el crecimiento poblacional sin considerar la desigualdad. No podemos diseñar políticas de bienestar sin entender cómo la expansión urbana presiona la estructura ecológica. No podemos proyectar el futuro mirando variables aisladas.

**[Hacer pausa, dejar que la idea penetre]**

Las ciudades son SISTEMAS COMPLEJOS con retroalimentaciones no lineales. Y esta complejidad nos ha llevado históricamente a dos extremos problemáticos:

Por un lado, modelos conceptuales ricos pero sin poder predictivo cuantitativo. Por otro, análisis estadísticos que capturan correlaciones pero ignoran las causas estructurales.

**[DIAPOSITIVA 3 - Problema específico de Bogotá]**

**Pitch (Parte 2 - La Situación Crítica de Bogotá):**

Veamos los números de Bogotá que nos motivaron:

**Población:** De 6.8 millones en 2012 a 8 millones en 2024. Un crecimiento del 17% en 12 años. ¿Hasta dónde puede crecer? ¿Cuál es la capacidad de carga real?

**Desigualdad:** Gini de 0.52, uno de los más altos de América Latina. Y aquí está el hallazgo inquietante de estudios previos: la alta desigualdad NO solo es un problema social, es un FRENO al crecimiento poblacional. La gente migra, las familias reducen su tamaño, la ciudad expulsa población cuando la brecha se hace insostenible.

**Bienestar:** Índices estancados cerca de 99 sobre 100 (aparentemente alto), pero con volatilidad creciente. ¿Qué significa un índice alto si la desigualdad lo está erosionando constantemente?

**Territorio:** La huella urbana creció de 38,000 a 45,000 hectáreas. Mientras tanto, la estructura ecológica protegida se mantiene en ~15,000 hectáreas, una presión ambiental evidente.

**[Transición crítica - bajar el tono, volverse más serio]**

Estos números NO evolucionan independientemente. Déjenme mostrarles las retroalimentaciones que descubrimos:

**[DIAPOSITIVA 4 - Retroalimentaciones del sistema]**

**Pitch (Parte 3 - Las Retroalimentaciones Ocultas):**

**Primera retroalimentación - Desigualdad frena Población:**
Cuando la desigualdad aumenta (F↑), la población tiende a estancarse o decrecer (P↓). No es obvio, pero los datos lo confirman: en años de Gini alto, las tasas de crecimiento poblacional caen. ¿Por qué? Migración hacia ciudades intermedias, reducción de natalidad en hogares vulnerables, expulsión económica.

Este es el parámetro gamma_F que estimaremos: resultó ser -1.69. Negativo y fuerte. La desigualdad es un FRENO poblacional significativo.

**Segunda retroalimentación - Bienestar influye en Desigualdad (pero débilmente):**
Teoría económica dice que mejorar el bienestar promedio debería reducir la desigualdad (mediante políticas redistributivas implícitas). Pero encontramos que beta_F = 0.01. Casi cero.

Esto es un hallazgo CRÍTICO: Bogotá tiene una desconexión entre bienestar agregado y reducción de desigualdad. Mejoramos índices promedio, pero la brecha permanece.

**Tercera retroalimentación - Desigualdad penaliza Bienestar:**
Esta es la más fuerte: delta_W = 1.0. Por cada unidad que aumenta la desigualdad, el bienestar cae proporcionalmente. Es una relación 1 a 1, casi determinista.

Implicación: NO podemos mejorar el bienestar sostenidamente sin atacar la desigualdad. Son vasos comunicantes.

**Cuarta retroalimentación - Población presiona Bienestar:**
alpha_W = -0.01. Ligeramente negativo. A mayor población, leve caída en bienestar (congestión, presión sobre servicios). No es dramático, pero existe.

**Quinta retroalimentación - Infraestructura afecta Bienestar:**
beta_W = -0.08. Contraintuitivo: la suma de huella urbana y estructura ecológica (U+E) tiene efecto NEGATIVO en bienestar. Interpretación: posible multicolinealidad o captura de efectos de congestión y fragmentación ecológica.

**[Pausa estratégica - hacer contacto visual con el jurado]**

**Pitch (Parte 4 - Por Qué Este Modelo Es Necesario AHORA):**

Entonces, ¿por qué este proyecto es urgente y relevante?

**Razón 1 - Vacío Metodológico:**
No existe en la literatura un modelo matemático CALIBRADO con datos reales que integre población, desigualdad y bienestar para ciudades latinoamericanas. Los modelos de System Dynamics de Forrester (1969) son cualitativos. Los modelos econométricos urbanos capturan correlaciones pero no dinámicas.

Este trabajo llena ese vacío: EDOs calibradas, validación cuantitativa (R² de 0.96 en población), análisis de estabilidad matemática (teorema de Hartman-Grobman).

**Razón 2 - Herramienta de Política Pública:**
Bogotá gasta ~50 billones de pesos anuales en presupuesto. Las decisiones sobre vivienda, infraestructura, programas sociales se hacen con proyecciones lineales o escenarios ad-hoc.

Este modelo permite:

- Proyectar escenarios "qué pasaría si expandimos K (capacidad) en 10%"
- Evaluar "qué pasaría si logramos aumentar beta_F (vínculo bienestar-desigualdad) de 0.01 a 0.05"
- Identificar puntos de inflexión ANTES de que ocurran

Es una herramienta de soporte a decisión basada en evidencia, no en intuición.

**Razón 3 - Replicabilidad Regional:**
El marco es generalizable. Si funciona en Bogotá, puede calibrarse para Medellín, Cali, Ciudad de México, Lima, Buenos Aires.

Imaginemos una red de modelos urbanos acoplados en América Latina, compartiendo metodología, comparando parámetros. Podríamos identificar patrones regionales: ¿Todas las capitales tienen gamma_F negativo? ¿Beta_F es universalmente bajo en la región?

**Razón 4 - Fundamento Teórico Riguroso:**
No es un modelo "caja negra" de machine learning. Cada ecuación tiene justificación física:

- Ecuación P: biología poblacional (Verhulst 1838) + efectos socioeconómicos
- Ecuación F: relajación estructural + influencia de bienestar (economía del desarrollo)
- Ecuación W: función de producción urbana + externalidades

Los parámetros son interpretables. No solo predecimos, ENTENDEMOS.

**[DIAPOSITIVA 4 final - Transición a objetivos]**

**Pitch (Parte 5 - El Puente hacia la Solución):**

Por todo esto, el objetivo de este trabajo NO es solo académico. Es construir una herramienta que:

1. **Explique** por qué Bogotá llegó a donde está (validación histórica con R² > 0.9 en población)
2. **Prediga** hacia dónde va bajo escenarios plausibles (proyecciones a 5 años)
3. **Prescriba** qué palancas mover para cambiar el rumbo (análisis de sensibilidad paramétrico)

En los próximos 15 minutos les mostraré:

- Cómo formulé matemáticamente estas retroalimentaciones
- Cómo calibré 10 parámetros con 13 años de datos
- Qué descubrí sobre la estabilidad del sistema (hay un atractor único, predecible)
- Qué escenarios proyectamos para 2024-2029
- Y qué recomendaciones concretas emergen para política pública

Pero antes de entrar en ecuaciones, déjenme mostrarles el objetivo general y los específicos...

**[TRANSICIÓN NATURAL A DIAPOSITIVA 5 - Objetivos]**

---

## ELEMENTOS RETÓRICOS CLAVE EN ESTE PITCH AMPLIADO:

### 1. **Gancho Emocional-Racional:**

- Comienza con pregunta (involucra a la audiencia)
- Datos impactantes (8 de cada 10, 8 millones de habitantes)
- Contraste (conceptual vs. cuantitativo, correlación vs. causalidad)

### 2. **Estructura de Problema-Solución:**

- Problema global → Problema regional → Problema específico Bogotá
- Retroalimentaciones como "descubrimientos" (genera curiosidad)
- Vacío metodológico (justifica la contribución)

### 3. **Lenguaje Visual:**

- "Laboratorio viviente"
- "Vasos comunicantes"
- "Freno poblacional"
- "Desconexión entre bienestar y desigualdad"

### 4. **Anticipación de Preguntas:**

- "¿Por qué un modelo matemático?" → Porque las políticas actuales son ad-hoc
- "¿Por qué Bogotá?" → Porque es representativa y tiene datos
- "¿Para qué sirve?" → Soporte a decisión, replicable

### 5. **Transiciones Estratégicas:**

- Pausas después de datos críticos
- "Aquí está el hallazgo inquietante..."
- "Déjenme mostrarles..."
- "Por todo esto..."

---

## TIMING SUGERIDO PARA ESTAS 3 DIAPOSITIVAS:

- **Diapositiva 2 (Desafío urbano):** 1:30 min
- **Diapositiva 3 (Situación Bogotá):** 1:30 min
- **Diapositiva 4 (Retroalimentaciones + Por qué es necesario):** 2:30 min
- **Transición a objetivos:** 0:30 min

**Total:** 6 minutos (30% del tiempo total, justificado por la importancia del contexto)

---

## AJUSTE DEL TIMING TOTAL DE LA PRESENTACIÓN:

Con este contexto ampliado:

- **Contexto/Motivación:** 6 min (ampliado)
- **Objetivos:** 1 min
- **Marco teórico/Variables:** 1.5 min (reducido ligeramente)
- **Sistema EDOs:** 1.5 min
- **Metodología/Calibración:** 2 min
- **Resultados (parámetros + validación):** 2 min
- **Análisis de fase/Estabilidad:** 2 min
- **Proyecciones:** 1.5 min
- **Discusión/Limitaciones:** 1.5 min
- **Conclusiones/Recomendaciones:** 2 min

**Total:** 21 minutos + margen para preguntas

---

## DOCUMENTO DE APOYO VISUAL (si usas pizarra o puntero):

Durante el pitch de retroalimentaciones, puedes dibujar rápidamente:

```
    ↗ P (población)
   /     ↘
  /       ↘ (alpha_W: -0.01)
 /         ↘
F ← (beta_F: 0.01) ← W (bienestar)
 \         ↑
  \       / (delta_W: -1.0)
   ↘    /
    ↘ /
   (gamma_F: -1.69)
```

Y señalar: "Miren este ciclo: F penaliza a P y a W, pero W apenas afecta a F. Es un sistema desequilibrado hacia la desigualdad."

---

¿Quieres que desarrolle algún otro pitch específico con este nivel de detalle, o ajuste el timing de otras secciones?

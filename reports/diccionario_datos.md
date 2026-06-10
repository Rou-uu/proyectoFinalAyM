# Diccionario de datos - C5 incidentes viales (CDMX 2022-2024)

**Proyecto Final - Almacenes y Mineria de Datos.** Profesora: Jessica Santizo Galicia.
Ayudantes: Diego Antonio Villalba Gonzalez y Ares Gael Castro Romero. Fecha: 2026-06.

Documenta cada variable, su significado, los valores que puede tomar y su rol
(target / feature / excluida / id). Dataset de trabajo: data/c5_listo.parquet
(289,885 filas, sin duplicados y sin el ano-ruido 2021).

| Variable | Tipo | Significado | Valores posibles / rango | Rol |
|---|---|---|---|---|
| `folio` | texto (id) | Identificador unico del reporte. | Cadena tipo C2C/20211229/00212. | id (no feature) |
| `FECHA_CREACION` | datetime | Fecha-hora de creacion del reporte. | 2022-01 a 2024-02 (tras depurar). | excluida (id temporal) |
| `codigo_cierre` | categorica | Como cerro el reporte. Base del target. | A=Afirmativo (real), D=Duplicado (real, ya reportado), F=Falso, I=Informativo. | excluida (base del target) |
| `CIERRE_DESC` | texto | Descripcion legible de codigo_cierre. | Texto libre por codigo. | excluida (descriptivo) |
| `REAL` | entero 0/1 | TARGET: el reporte es real (afirmativo)? | 1 si codigo_cierre=='A'; 0 si F/I. Definido tras quitar D. | target |
| `HORA` | entero | Hora del dia del reporte. | 0-23. | feature (num) |
| `MES` | entero | Mes del reporte. | 1-12. | feature (num) |
| `ANIO` | entero | Ano del reporte. | 2022-2024. | feature (num) |
| `latitud` | float | Latitud del incidente (CDMX). | ~19.09-19.63. | feature (num / clustering) |
| `longitud` | float | Longitud del incidente (CDMX). | ~-99.35 a -98.95. | feature (num / clustering) |
| `tipo_incidente_c4` | cat (7) | Tipo grueso de incidente. | Accidente, Lesionado, Cadaver, Detencion ciudadana, Mi Calle, Sismo, Mi Taxi. | feature |
| `incidente_c4` | cat (16) | Subtipo detallado del incidente. | Choque sin/con lesionados, Atropellado, Motociclista, Volcadura, Ciclista, Persona atrapada/desbarrancada, Vehiculo atrapado/varado, Choque con prensados, Accidente automovilistico, Incidente de transito, Vehiculo desbarrancado, Otros, Monopatin, Ferroviario, Persona atropellada. | feature |
| `tipo_entrada` | cat (~9) | Canal por el que entro el reporte. | LLAMADA DEL 911 (mayoria), BOTON DE AUXILIO, RADIO, CAMARA, REDES, APLICATIVOS, LLAMADA APP911, SOS MUJERES *765, LECTOR DE PLACAS. (3 nulos.) | feature |
| `alcaldia_catalogo` | cat (16) | Alcaldia de CDMX del incidente. | Las 16 alcaldias (Iztapalapa, Gustavo A. Madero, Cuauhtemoc, ... Milpa Alta). (366 nulos.) | feature |
| `dia_semana` | cat (7) | Dia de la semana del reporte. | Lunes...Domingo. | feature |
| `FRANJA` | cat (4) | Franja horaria (derivada de HORA). | Madrugada, Mañana, Tarde, Noche. | feature |
| `clas_con_f_alarma` | categorica | Clasificacion operativa (con falsa alarma). | URGENCIAS MEDICAS, EMERGENCIA, DELITO, FALSA ALARMA, INCIDENTES EXTERNOS. | excluida (fuga fina) |
| `TIEMPO_CIERRE_MIN` | float (min) | Tiempo hasta cerrar el reporte. | Minutos > 0 (2 nulos). | excluida (fuga gruesa) |

## Notas de rol

- TARGET (REAL): se define tras ELIMINAR los duplicados D. Un cierre D es un evento real
  ya reportado; su deduplicacion es tarea del sistema de despacho, no del modelo de
  veracidad. Entre el resto, REAL=1 si 'A' (afirmativo), 0 si 'F'/'I'.
- Excluidas por fuga: clas_con_f_alarma (fuga fina: 'FALSA ALARMA' es clasificacion
  posterior al cierre) y TIEMPO_CIERRE_MIN (fuga gruesa: solo se conoce al cerrar).
- Excluidas por ser id o base del target: folio, FECHA_CREACION, codigo_cierre, CIERRE_DESC.

## Features del modelo
- Numericas: HORA, MES, ANIO, latitud, longitud.
- Categoricas: tipo_incidente_c4, incidente_c4, tipo_entrada, alcaldia_catalogo, dia_semana, FRANJA.

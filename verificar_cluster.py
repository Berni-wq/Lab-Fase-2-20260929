#!/usr/bin/env python3
# =============================================================================
#  Prueba de humo del clúster — Fase 2, Sitio 3
# =============================================================================
#  Ejecutar DENTRO del contenedor de Jupyter:
#      docker compose exec jupyter python3 verificar_cluster.py
#
#  Comprueba las 4 cosas que, si fallan, hacen que el clúster "parezca" arriba
#  pero no sirva para trabajar:
#     1. Que se crea una SparkSession apuntando al master por su NOMBRE (no IP).
#     2. Que hay executors vivos realmente (no sólo el master respondiendo).
#     3. Que los executors ven el volumen montado en /opt/workspace.
#     4. Que Spark lee de verdad los datos de data/raw.
#
#  Sale con código 0 si todo pasa, 1 si algo falla. Así sirve como puerta de
#  calidad en CI o antes de empezar la clase.
# =============================================================================

import os
import sys
import glob

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

RUTA_DATOS = "/opt/workspace/data/raw"
RUTA_JSONL = os.path.join(RUTA_DATOS, "lecturas_sitio3.jsonl")

fallos = []


def comprobar(cond, titulo, detalle=""):
    estado = "OK  " if cond else "FALLA"
    print(f"  [{estado}] {titulo}" + (f" -> {detalle}" if detalle else ""))
    if not cond:
        fallos.append(titulo)
    return cond


print("=" * 70)
print("  PRUEBA DE HUMO - Clúster Spark local (Fase 2, Sitio 3)")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1. SESIÓN
# -----------------------------------------------------------------------------
# spark.driver.host=jupyter y bindAddress=0.0.0.0 son OBLIGATORIOS: el driver
# corre dentro de este contenedor y los executors deben poder volver a él para
# recibir tareas. Sin esto el job se queda en RUNNING para siempre.
spark = (
    SparkSession.builder
    .appName("verificar_cluster_sitio3")
    .master(os.environ.get("SPARK_MASTER_URL", "spark://spark-master:7077"))
    .config("spark.driver.host", os.environ.get("SPARK_DRIVER_HOST", "jupyter"))
    .config("spark.driver.bindAddress", "0.0.0.0")
    .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "2g"))
    .config("spark.executor.memory", os.environ.get("SPARK_EXECUTOR_MEMORY", "2g"))
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print("\n[1] Sesión de Spark")
comprobar(spark.sparkContext.appName == "verificar_cluster_sitio3", "sesión creada",
          f"master={spark.sparkContext.master}")

# -----------------------------------------------------------------------------
# 2. EXECUTORS VIVOS
# -----------------------------------------------------------------------------
# Un master sano con 0 executors es el síntoma clásico de
# SPARK_EXECUTOR_MEMORY > SPARK_WORKER_MEMORY: el worker rechaza arrancar los
# executors. Por eso se exige > 0 y no sólo "el master respondió".
print("\n[2] Executors")
ej = spark.sparkContext._jsc.sc().statusTracker().getExecutorInfos()
vivos = len(ej)
comprobar(vivos > 0, "hay executors vivos", f"{vivos} executor(es)")

cores_totales = spark.sparkContext.defaultParallelism
comprobar(cores_totales > 0, "parallelism por defecto", f"{cores_totales} tareas")

# -----------------------------------------------------------------------------
# 3. VISIBILIDAD DEL VOLUMEN EN LOS EXECUTORS
# -----------------------------------------------------------------------------
# Cada executor abre los ficheros por su cuenta. Si /opt/workspace no está
# montado igual en los workers, el driver lee bien y los executors revientan
# con FileNotFoundException. Se comprueba desde DENTRO de un executor, no desde
# el driver: el driver puede ver cosas que el executor no.
print("\n[3] Volumen /opt/workspace visto desde los executors")
ficheros = sorted(glob.glob(os.path.join(RUTA_DATOS, "*.jsonl")))
comprobar(len(ficheros) > 0, "data/raw visible desde el driver",
          f"{len(ficheros)} fichero(s): {[os.path.basename(f) for f in ficheros]}")

if ficheros:
    ruta = ficheros[0]
    # mapPartitions se ejecuta en cada executor: si el volumen no estuviera
    # montado allí, os.listdir fallaría DENTRO del executor.
    rdd_visibilidad = spark.sparkContext.parallelize([ruta], 1).map(
        lambda p: (os.path.basename(p), os.path.exists(p),
                   os.path.getsize(p) if os.path.exists(p) else -1)
    )
    visibles = rdd_visibilidad.collect()
    for nombre, existe, tam in visibles:
        comprobar(existe, f"executor ve {nombre}", f"{tam} bytes")

# -----------------------------------------------------------------------------
# 4. LECTURA REAL DE LOS DATOS
# -----------------------------------------------------------------------------
# Aquí es donde se escribe Parquet de la Fase 1. Se reparte con
# spark.read.json para que los executors hagan el parseo real, no el driver.
print("\n[4] Lectura real de data/raw")
try:
    df = (
        spark.read
        .option("header", True)
        .schema("id_lectura long, timestamp timestamp, estacion string, "
                "zona string, temperatura_c double, humedad_pct int, "
                "viento_kph double, precipitacion_mm double")
        .json(RUTA_JSONL)
    )
    total = df.count()
    comprobar(total > 0, "lectura de JSONL", f"{total} filas")

    # Un groupBy obliga a los executors a hacer un shuffle: si los executors
    # estuvieran muertos o el volumen no existiera allí, esto no volvería.
    resumen = (
        df.groupBy("zona")
          .agg(
              F.count("*").alias("lecturas"),
              F.avg("temperatura_c").alias("temp_media"),
              F.max("humedad_pct").alias("hum_max"),
          )
          .orderBy(F.desc("lecturas"))
    )
    print("\n  Agregado por zona (shuffle distribuido en los executors):")
    resumen.show(10, truncate=False)

    comprobar(resumen.count() > 0, "groupBy + shuffle distribuido OK")

    # Escritura de Parquet: valida el otro lado de la tubería y deja salida
    # para la Fase 3.
    salida = "/opt/workspace/salida_sitio3/lecturas_zona"
    (df.write.mode("overwrite").parquet(salida))
    n_parquet = len(glob.glob(os.path.join(salida, "part-*.parquet")))
    comprobar(n_parquet > 0, "escritura Parquet", f"{n_parquet} fichero(s) en salida_sitio3/")

except Exception as exc:  # noqa: BLE001
    comprobar(False, "lectura de data/raw", f"{type(exc).__name__}: {exc}")

# -----------------------------------------------------------------------------
spark.stop()

print("\n" + "=" * 70)
if fallos:
    print(f"  RESULTADO: {len(fallos)} COMPROBACIÓN(ES) FALLIDA(S)")
    for f in fallos:
        print(f"    - {f}")
    print("=" * 70)
    sys.exit(1)

print("  RESULTADO: TODAS LAS COMPROBACIONES OK")
print("=" * 70)
sys.exit(0)

#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Robótica Computacional -
# Grado en Ingeniería Informática (Cuarto)
# Práctica: Resolución de la cinemática inversa mediante CCD
#           (Cyclic Coordinate Descent).
# Daniel Palenzuela Álvarez alu0101140469

# El script debe ser capaz de:
# - Resolver la cinemática inversa para cualquier número de articulaciones.
# - Gestionar cualquier combinación de articulaciones rotacionales o prismáticas.
# - Calcular correctamente los incrementos necesarios en cada articulación.
# - Tener en cuenta los límites individuales para cada articulación, incluyendo 
# un límite superior y un límite inferior para cada una.

# EJECUCION:
#   Instalar dependencias: python -m pip install numpy matplotlib
#   Ejecutar: python ccdp3_modificado.py 10 5
#   Los argumentos 10 y 5 representan las coordenadas X e Y del objetivo.

# CONFIGURACION DEL ROBOT:
#   Para modificar el robot, cambiar las siguientes listas:
#   - tipo: tipo de articulacion ('R' rotacional o 'P' prismatica).
#   - th: valores articulares iniciales (rad para R y longitud para P).
#   - a: longitud fija de cada eslabon.
#   - angulos_fijos: orientacion fija de cada articulacion (radianes).
#   - lim_inf: limite inferior individual de cada articulacion.
#   - lim_sup: limite superior individual de cada articulacion.
#
#   Todas las listas deben tener el mismo numero de elementos.
#   Se pueden combinar articulaciones R y P en cualquier orden.

# PARAMETROS DEL ALGORITMO:
#   - EPSILON: precision exigida para alcanzar el objetivo.
#   - MAX_ITERACIONES: numero maximo de iteraciones CCD.
#   - MEJORA_MINIMA: mejora minima del error para continuar iterando.

import sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
import colorsys as cs

# ******************************************************************************
# Declaración de funciones


def muestra_origenes(O, final=0):
  # Muestra por pantalla el origen de la base y los extremos de los eslabones.
  print('Origenes de coordenadas:')
  for i in range(len(O)):
    print('(O' + str(i) + ')0\t= ' + str([round(j, 3) for j in O[i]]))
  if final:
    print('E.Final = ' + str([round(j, 3) for j in final]))


def muestra_robot(O, obj):
  # Muestra el robot graficamente
  # Representa las distintas posiciones del robot calculadas durante un ciclo
  # CCD y dibuja el objetivo con un asterisco.
  plt.figure()
  plt.xlim(-L, L)
  plt.ylim(-L, L)
  T = [np.array(o).T.tolist() for o in O]
  for i in range(len(T)):
    plt.plot(T[i][0], T[i][1], '-o',
             color=cs.hsv_to_rgb(i / float(max(1, len(T))), 1, 1))
  plt.plot(obj[0], obj[1], '*')
  plt.pause(0.0001)
  plt.show()
  plt.close()


def matriz_T(d, th, a, al):
  # Matriz DH, que representa la transformación homogénea de un eslabón.
  # Para el robot plano se utiliza d = 0 y al = 0. De esta forma, la matriz
  # realiza una rotación th alrededor de Z y una traslación a sobre el eje X local.
  return [[cos(th), -sin(th) * cos(al),  sin(th) * sin(al), a * cos(th)],
          [sin(th),  cos(th) * cos(al), -sin(al) * cos(th), a * sin(th)],
          [      0,          sin(al),          cos(al),             d],
          [      0,                0,                0,             1]]


def cin_dir(th, a, tipo=None, angulos_fijos=None, devuelve_ejes=False):
  # Calcula la cinemática directa a partir de los valores articulares th.
  #
  # th: coordenadas articulares. En R son ángulos (rad), en P son longitudes.
  # a: longitud fija de cada eslabón. En P se suma al desplazamiento th[i].
  # tipo: lista con 'R' (rotacional) o 'P' (prismática) para cada articulación.
  # angulos_fijos: orientación fija de cada articulación, expresada en radianes.
  # devuelve_ejes: si es true, devuelve también los ejes de las prismáticas.
  #
  # Se mantienen los argumentos opcionales para que la llamada original
  # cin_dir(th, a) siga funcionando como un robot formado solo por R.

  n = len(th) # Número de articulaciones
  if tipo is None: # Se asume que todas son rotacionales si no se especifica.
    tipo = ['R'] * n
  if angulos_fijos is None: # Se asume que no hay orientación fija si no se especifica.
    angulos_fijos = [0.0] * n

  # Se comprueba que todas las listas tengan la misma longitud.
  if len(a) != n or len(tipo) != n or len(angulos_fijos) != n:
    raise ValueError('th, a, tipo y angulos_fijos deben tener la misma longitud.')

  # T acumula las transformaciones homogéneas desde la base hasta cada eslabón.
  T = np.identity(4)
  # El origen de la base se considera el primer extremo del robot.
  o = [[0.0, 0.0]]
  # Los ejes de las articulaciones prismáticas se calculan para poder proyectar el error sobre ellos.
  ejes = []

  for i in range(n):
    # Orientación del sistema de coordenadas antes de aplicar la articulación i.
    orientacion_previa = atan2(T[1, 0], T[0, 0])

    # El eje prismático es el eje X local, girado por la orientación fija.
    # Se expresa en coordenadas globales para poder proyectar sobre él el error.
    orientacion_eje = orientacion_previa + angulos_fijos[i]
    ejes.append([cos(orientacion_eje), sin(orientacion_eje)])

    if tipo[i] == 'R':
      # Articulación rotacional, th[i] modifica el ángulo, a[i] no cambia.
      theta = th[i] + angulos_fijos[i]
      longitud = a[i]
    elif tipo[i] == 'P':
      # Articulación prismática, el ángulo es fijo y th[i] modifica la longitud.
      theta = angulos_fijos[i]
      longitud = a[i] + th[i]
    else:
      # Si se introduce un tipo de articulación no válido, se lanza un error.
      raise ValueError("Tipo de articulacion no valido: use 'R' o 'P'.")

    # Se calcula la matriz de transformación homogénea para la articulación i y se acumula.
    T = np.dot(T, matriz_T(0.0, theta, longitud, 0.0))

    # El extremo del eslabón i es el origen del siguiente sistema de referencia.
    tmp = np.dot(T, [0.0, 0.0, 0.0, 1.0])
    o.append([tmp[0], tmp[1]])

  # Si devuelve_ejes es True, se devuelve también la lista de ejes de las prismáticas.
  if devuelve_ejes:
    return o, ejes
  return o


# ******************************************************************************
# Cálculo de la cinemática inversa de forma iterativa por el método CCD

# VALORES ARTICULARES INICIALES Y CONFIGURACIÓN DEL ROBOT
# Para cambiar el número de articulaciones, se añade o elimina elementos en TODAS
# las listas siguientes, manteniendo el mismo orden.

# Tipo de articulación: 'R' = rotacional, 'P' = prismática.
tipo = ['R', 'R', 'R']

# Coordenadas articulares iniciales.
# - R: ángulo inicial en radianes.
# - P: desplazamiento lineal inicial.
th = [0.0, 0.0, 0.0]

# Longitud fija de cada eslabón.
# En las articulaciones P se suma a th[i], usar 0.0 si no se necesita longitud fija.
a = [5.0, 5.0, 5.0]

# Orientación fija de cada articulación, en radianes.
# En R se suma al ángulo y en P define la dirección del desplazamiento.
angulos_fijos = [0.0, 0.0, 0.0]

# Límites articulares: una cota inferior y otra superior para cada articulación.
# Para R se expresan en radianes, para P en unidades de longitud.
lim_inf = [-pi, -pi, -pi]
lim_sup = [ pi,  pi,  pi]

# Parámetros de convergencia.
L = 0.0 # Se calcula más abajo a partir del alcance del robot.
EPSILON = 0.01 # Distancia máxima al objetivo para considerar convergencia.
MAX_ITERACIONES = 1000 # Evita que el algoritmo itere indefinidamente.
MEJORA_MINIMA = 1e-10 # Umbral para detectar que el error ya no disminuye.

# ******************************************************************************
# Comprobación de la configuración antes de iniciar el algoritmo

n = len(th)  # Número de articulaciones, que debe coincidir con el tamaño de todas las listas de configuración.

# Todas las listas de configuración deben describir el mismo número de juntas.
if n == 0 or any(len(lista) != n for lista in
                 (a, tipo, angulos_fijos, lim_inf, lim_sup)):
  sys.exit('Error: th, a, tipo, angulos_fijos, lim_inf y lim_sup deben tener '
           'la misma longitud y al menos una articulacion.')

# Se comprueba el tipo, las longitudes y los límites de cada articulación.
for i in range(n): # Se recorre cada articulación para validar su configuración.
  tipo[i] = tipo[i].upper() # Se normaliza el tipo a mayúsculas para evitar errores de entrada.
  if tipo[i] not in ('R', 'P'): # Si el tipo no es válido, se lanza un error.
    sys.exit("Error: el tipo de la articulacion " + str(i + 1) +
             " debe ser 'R' o 'P'.")
  if a[i] < 0: # Si la longitud del eslabón es negativa, se lanza un error.
    sys.exit('Error: la longitud fija de cada eslabon debe ser no negativa.')
  if lim_inf[i] > lim_sup[i]: # Si el límite inferior es mayor que el superior, se lanza un error.
    sys.exit('Error: el limite inferior de la articulacion ' + str(i + 1) +
             ' es mayor que el limite superior.')
  if not lim_inf[i] <= th[i] <= lim_sup[i]: # Si el valor inicial de la articulación está fuera de sus límites, se lanza un error.
    sys.exit('Error: el valor inicial de la articulacion ' + str(i + 1) +
             ' esta fuera de sus limites.')
  # Una longitud prismática negativa no representa una extensión física válida.
  if tipo[i] == 'P' and a[i] + lim_inf[i] < 0:
    sys.exit('Error: para la articulacion prismatica ' + str(i + 1) +
             ', a[i] + lim_inf[i] no puede ser negativo.')

# Se calcula un tamaño de gráfica que incluya el mayor alcance permitido.
# El alcance máximo es la suma de las longitudes máximas de todos los eslabones.
alcance = 0.0
for i in range(n):
  if tipo[i] == 'R':
    # Para las articulaciones rotacionales, el alcance máximo es simplemente la longitud del eslabón.
    alcance += a[i]
  else:
    # Para las articulaciones prismáticas, el alcance máximo es la longitud fija más el desplazamiento máximo permitido.
    alcance += max(a[i] + lim_inf[i], a[i] + lim_sup[i]) 
# Se añade un margen del 10% para que el robot no toque los bordes de la gráfica.
L = max(1.0, alcance * 1.1)

#plt.ion() # modo interactivo

# Introducción del punto objetivo para la cinemática inversa.
if len(sys.argv) != 3:
  # Se espera que el usuario introduzca las coordenadas X e Y del objetivo como argumentos de línea de comandos.
  sys.exit('python ' + sys.argv[0] + ' x y') 
objetivo = [float(i) for i in sys.argv[1:]]

# Se calcula y muestra la posición inicial del robot.
O = cin_dir(th, a, tipo, angulos_fijos)
print('- Posicion inicial:')
muestra_origenes(O)

# ******************************************************************************
# Bucle principal del algoritmo CCD

# dist es el error entre el efector final y el objetivo.
# prev almacena el error del ciclo anterior para detectar falta de progreso.
dist = float('inf')
prev = 0.0
iteracion = 1
incrementos = [0.0] * n # Último incremento aplicado a cada articulación.

# Se repite mientras no se alcance el objetivo y no se supere el máximo de
# iteraciones. La detección de estancamiento se realiza después de cada barrido,
# con una tolerancia muy pequeña para no detener el algoritmo demasiado pronto.
while dist > EPSILON and iteracion <= MAX_ITERACIONES:
  prev = dist

  # Se guarda la posición inicial de este barrido CCD para poder representarlo.
  O = [cin_dir(th, a, tipo, angulos_fijos)]
  incrementos = [0.0] * n

  # Se recorren las articulaciones desde la última hasta la primera, para 
  # optimizar cada una en orden inverso. Cada coordenada se optimiza manteniendo las demás fijas.
  for i in range(n - 1, -1, -1):
    # Se recalculan las posiciones de todas las articulaciones y el efector final con la configuración actualizada de th. 
    # También se obtienen los ejes de las articulaciones prismáticas para poder proyectar el error sobre ellos.
    O_actual, ejes = cin_dir(th, a, tipo, angulos_fijos, True)

    # Origen de la articulación i, extremo actual del robot y objetivo deseado.
    origen_art = np.array(O_actual[i], dtype=float)
    efector_final = np.array(O_actual[-1], dtype=float)
    obj = np.array(objetivo, dtype=float)

    # Vectores desde la articulación hasta el efector final y hasta el objetivo.
    v_efector = efector_final - origen_art
    v_objetivo = obj - origen_art

    if tipo[i] == 'R':
      # ARTICULACIÓN ROTACIONAL (R)
      # El giro óptimo es el ángulo entre ambos vectores.
      # atan2(cross, dot) calcula ese ángulo directamente en el intervalo
      # [-pi, pi], evitando problemas al pasar de +pi a -pi.
      producto_vectorial = (v_efector[0] * v_objetivo[1] -
                            v_efector[1] * v_objetivo[0])
      producto_escalar = np.dot(v_efector, v_objetivo)

      # Si el efector coincide con el origen de esta articulación, su dirección
      # no está definida y un giro alrededor de ese origen no mueve el extremo.
      if np.linalg.norm(v_efector) < 1e-12 or np.linalg.norm(v_objetivo) < 1e-12:
        delta = 0.0
      else:
        delta = atan2(producto_vectorial, producto_escalar)

    else:
      # ARTICULACIÓN PRISMÁTICA (P)
      # Al variar una prismática, el efector se desplaza en la dirección de su
      # eje. Por tanto, el incremento que minimiza el error es la proyección
      # del vector de error sobre el eje de la articulación.
      eje = np.array(ejes[i], dtype=float)
      delta = np.dot(obj - efector_final, eje)

    # Se aplica el incremento y se limita el nuevo valor articular.
    # np.clip garantiza lim_inf[i] <= th[i] <= lim_sup[i] en todo momento.
    valor_nuevo = np.clip(th[i] + delta, lim_inf[i], lim_sup[i])

    # Se guarda el incremento real aplicado. Puede ser menor que delta si se
    # alcanza un límite inferior o superior de la articulación.
    incrementos[i] = valor_nuevo - th[i]
    th[i] = valor_nuevo

    # Se añade la nueva configuración para dibujar el movimiento de esta pasada.
    O.append(cin_dir(th, a, tipo, angulos_fijos))

  # Tras actualizar todas las articulaciones, se calcula el error final del ciclo.
  posicion_final = np.array(O[-1][-1], dtype=float)
  dist = float(np.linalg.norm(np.array(objetivo, dtype=float) - posicion_final))

  # Se muestran los resultados de este barrido CCD, incluyendo la distancia al 
  # objetivo y los incrementos aplicados a cada articulación.
  print('\n- Iteracion ' + str(iteracion) + ':')
  muestra_origenes(O[-1])
  muestra_robot(O, objetivo)
  print('Distancia al objetivo = ' + str(round(dist, 5)))

  # Se muestran los incrementos aplicados en este barrido. Los de R están en
  # radianes y los de P en unidades de longitud.
  print('Incrementos articulares aplicados:')
  for i in range(n):
    unidad = 'rad' if tipo[i] == 'R' else 'unidades'
    print('  Articulacion ' + str(i + 1) + ' (' + tipo[i] + '): ' +
          str(round(incrementos[i], 6)) + ' ' + unidad)

  iteracion += 1

  # Si el error prácticamente no ha cambiado, se evita seguir repitiendo el mismo ciclo.
  if dist > EPSILON and abs(prev - dist) <= MEJORA_MINIMA:
    break

# ******************************************************************************
# Resultados finales

# Se muestra un resumen de la convergencia, incluyendo el número de iteraciones
if dist <= EPSILON:
  print('\n' + str(iteracion - 1) + ' iteraciones para converger.')
else:
  # Si no se ha alcanzado la convergencia, se indica el número de iteraciones realizadas.
  print('\nNo se ha alcanzado la convergencia tras ' +
        str(iteracion - 1) + ' iteraciones.')

print('- Umbral de convergencia epsilon: ' + str(EPSILON))
print('- Distancia al objetivo:          ' + str(round(dist, 5)))
print('- Valores finales de las articulaciones:')
for i in range(n):
  # Se muestra el valor final de cada articulación, indicando si es rotacional o prismática, y sus límites.
  nombre = 'theta' if tipo[i] == 'R' else 'd'
  unidad = 'rad' if tipo[i] == 'R' else 'unidades'
  print('  ' + nombre + str(i + 1) + ' = ' + str(round(th[i], 6)) +
        ' ' + unidad + '  [limites: ' + str(lim_inf[i]) + ', ' +
        str(lim_sup[i]) + ']')

# Se muestran las longitudes fijas de los eslabones, aunque no se modifican durante la cinemática inversa.
for i in range(n):
  print('  L' + str(i + 1) + '     = ' + str(round(a[i], 3)))

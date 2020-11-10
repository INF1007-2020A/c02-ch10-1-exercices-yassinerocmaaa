#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import wave
import struct
import math

import numpy as np
import scipy


SAMPLING_FREQ = 44100 # Hertz, taux d'échantillonnage standard des CD
SAMPLE_WIDTH = 16 # Échantillons de 16 bit
MAX_INT_SAMPLE_VALUE = 2**(SAMPLE_WIDTH-1) - 1


def merge_channels(channels):
	# Équivalent de :  [sample for samples in zip(*channels) for sample in samples]
	return np.fromiter((sample for samples in zip(*channels) for sample in samples), float)

def separate_channels(samples, num_channels):
	# Équivalent de :  [samples[i::num_channels] for i in range(num_channels)]
	return np.fromiter((samples[i::num_channels] for i in range(num_channels)), float)

def generate_sample_time_points(duration):
	# Générer un tableau de points temporels également espacés en seconde. On a SAMPLING_FREQ points par seconde.
	return np.arange(0, duration * SAMPLING_FREQ) / SAMPLING_FREQ

def sine(freq, amplitude, duration):
	# Générer une onde sinusoïdale à partir de la fréquence et de l'amplitude donnée, sur le temps demandé et considérant le taux d'échantillonnage.
	# Formule de la valeur y d'une onde sinusoïdale à l'angle x en fonction de sa fréquence F et de son amplitude A :
	# y = A * sin(F * x), où x est en radian.
	# Si on veut le x qui correspond au moment t, on peut dire que 2π représente une seconde, donc x = t * 2π,
	# Or t est en secondes, donc t = i / nb_échantillons_par_secondes, où i est le numéro d'échantillon.

	# y = A * sin(F * 2π*t)
	time_points = generate_sample_time_points(duration)
	return amplitude * np.sin(freq * 2 * np.pi * time_points)

def square(freq, amplitude, duration):
	# Générer une onde carrée d'une fréquence et amplitude donnée.
	# y = A * sgn(sin(F * 2π*t))
	return amplitude * np.sign(sine(freq, 1, duration))

def sine_with_overtones(root_freq, amplitude, overtones, duration):
	# Générer une onde sinusoïdale avec ses harmoniques. Le paramètre overtones est une liste de tuple où le premier élément est le multiple de la fondamentale et le deuxième élément est l'amplitude relative de l'harmonique.
	# On bâtit un signal avec la fondamentale
	signal = sine(root_freq, amplitude, duration)
	# Pour chaque harmonique (overtone en Anglais), on a un facteur de fréquence et un facteur d'amplitude :
	for freq_factor, amp_factor in overtones:
		# Construire le signal de l'harmonique en appliquant les deux facteurs.
		overtone = sine(root_freq * freq_factor, amplitude * amp_factor, duration)
		# Ajouter l'harmonique au signal complet.
		np.add(signal, overtone, out=signal)
	return signal

def normalize(samples, norm_target):
	# Normalisez un signal à l'amplitude donnée
	# 1. il faut trouver l'échantillon le plus haut en valeur absolue
	abs_samples = np.abs(samples)
	max_sample = max(abs_samples)
	# 2. Calcule coefficient entre échantillon max et la cible
	coeff = norm_target / max_sample
	# 3. Applique mon coefficient
	normalized_samples = coeff * samples
	return normalized_samples

def convert_to_bytes(samples):
	# Convertir les échantillons en tableau de bytes en les convertissant en entiers 16 bits.
	# Les échantillons en entrée sont entre -1 et 1, nous voulons les mettre entre -MAX_INT_SAMPLE_VALUE et MAX_INT_SAMPLE_VALUE
	# Juste pour être certain de ne pas avoir de problème, on doit clamper les valeurs d'entrée entre -1 et 1.
	# 1. Limiter (ou clamp/clip) les échantillons entre -1 et 1
	clipped = np.clip(samples, -1, 1)
	# 2. convertir en entier 16-bit signés
	int_samples = (clipped * MAX_INT_SAMPLE_VALUE).astype("<i2")
	# 3. convertir en bytes
	sample_bytes = int_samples.tobytes()
	# Retourne le tout.
	return sample_bytes

def convert_to_samples(bytes):
	# Faire l'opération inverse de convert_to_bytes, en convertissant des échantillons entier 16 bits en échantillons réels
	# 1. Convertir en numpy array du bon type (entier 16 bit signés)
	int_samples = np.frombuffer(bytes, dtype="<i2")
	# 2. Convertir en réel dans [-1, 1]
	samples = int_samples.astype(float) / MAX_INT_SAMPLE_VALUE
	return samples


def main():
	try:
		os.mkdir("output")
	except:
		pass

	foo = generate_sample_time_points(0.001)
	bar = sine(440, 0.5, 0.1)
	qux = normalize(bar, 1)

	with wave.open("output/perfect_fifth_panned.wav", "wb") as writer:
		writer.setnchannels(2)
		writer.setsampwidth(2)
		writer.setframerate(SAMPLING_FREQ)

		# On génére un la3 (220 Hz) et un mi4 (intonation juste, donc ratio de 3/2)
		samples1 = sine(220, 0.9, 30.0)
		samples2 = sine(220 * (3/2), 0.7, 30.0)

		# On met les samples dans des channels séparés (la à gauche, mi à droite)
		merged = merge_channels([samples1, samples2])
		data = convert_to_bytes(merged)

		writer.writeframes(data)

	with wave.open("output/major_chord.wav", "wb") as writer:
		writer.setnchannels(1)
		writer.setsampwidth(2)
		writer.setframerate(SAMPLING_FREQ)

		# Un accord majeur (racine, tierce, quinte, octave) en intonation juste
		root_freq = 220
		root = sine(root_freq, 1, 10.0)
		third = sine(root_freq * 5/4, 1, 10.0)
		fifth = sine(root_freq * 3/2, 1, 10.0)
		octave = sine(root_freq * 2, 1, 10.0)
		chord = normalize(root + third + fifth + octave, 0.89)

		writer.writeframes(convert_to_bytes(chord))

	with wave.open("output/overtones.wav", "wb") as writer:
		writer.setnchannels(1)
		writer.setsampwidth(2)
		writer.setframerate(SAMPLING_FREQ)

		samples = sine_with_overtones(220, 1, [(i, 0.15**(i-1)) for i in range(2, 16)], 10)
		samples = normalize(samples, 0.89)

		writer.writeframes(convert_to_bytes(samples))

if __name__ == "__main__":
	main()

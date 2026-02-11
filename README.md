# Exploring Our Seismic Data

This repository contains a growing collection of **worked, reproducible examples** demonstrating how to access and analyse seismic data from the University of Melbourne and partner networks using modern Python-based workflows.

The primary aim is educational and illustrative: to show *how* data can be accessed, processed, and interpreted, rather than to provide a monolithic analysis framework.

## Background

Seismic waveform and metadata from the University of Melbourne network are now available via an **FDSN web service**. This repository is intended to provide clear, executable examples that demonstrate how to:

- discover stations and waveforms via FDSN
- download and handle waveform data
- apply automated phase picking
- perform simple event-level analyses

These examples are designed to be easily adapted for teaching, training, and exploratory research.

## Repository structure  


Each `exXX/` directory is intended to be a **self-contained example**, including:
- a Jupyter notebook
- any required waveform, metadata, or auxiliary files
- minimal external dependencies beyond standard Python seismology tools

## Example 01 — Local earthquake location

**Example 01** demonstrates a simple end-to-end workflow for a local earthquake, including:

- accessing waveform and station metadata via FDSN
- automated phase picking
- inspection and export of picks
- preliminary event location

## Requirements

The examples primarily rely on standard Python scientific and seismological packages, including:

- `numpy`
- `matplotlib`
- `obspy`
- `seisbench`

Exact imports and setup are documented within each notebook.

## Intended audience

This repository is aimed at:
- students learning practical seismological workflows
- researchers new to FDSN-based data access
- collaborators looking for concrete, reproducible examples

The examples are intentionally explicit and verbose, prioritising clarity over conciseness.

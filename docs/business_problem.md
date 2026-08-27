# Business Problem

## Project

Steel Quality Prediction, Explainability & AI Analytics Copilot

## Business Context

Manufacturing quality engineers often work with large volumes of
structured inspection and production-quality data.

The challenge is not only identifying which defects occur, but also:

- understanding defect distribution,
- identifying samples with elevated predicted risk,
- predicting likely defect categories,
- understanding why a model produced a prediction,
- and accessing these results efficiently.

This project builds an AI-assisted quality analytics system that combines
structured data analytics, machine learning, explainability, and a
natural-language copilot.

## Unit of Analysis

Each observation represents a steel quality inspection sample described
by structured characteristics including geometry, surface and luminosity
measurements, steel type, plate thickness, and related numerical features.

## Target

The dataset contains seven defect indicators:

- Pastry
- Z_Scratch
- K_Scatch
- Stains
- Dirtiness
- Bumps
- Other_Faults

## Primary Machine Learning Task

The primary modeling task is:

**Single-label multiclass steel defect classification.**

Given the structured characteristics of an inspection sample, the model
predicts the most likely defect category.

Samples that do not contain exactly one positive defect label are treated
as data-quality exceptions and will be investigated separately rather
than silently converted into multiclass labels.

## Business User

The primary user is a:

**Manufacturing / Quality Engineer**

## Decision Supported

The system supports:

- quality inspection triage,
- defect analysis,
- prioritization of samples for further investigation,
- and model-assisted quality review.

## System Responsibilities

### SQL / PostgreSQL

Provides factual quality data and analytical results.

### Machine Learning

Predicts the most likely defect category.

### SHAP

Explains which features influenced the model prediction.

SHAP values describe model behavior and must not be interpreted as proof
of manufacturing causality.

### Local LLM Copilot

Uses function calling to select backend tools and summarize structured
results in natural language.

The LLM does not independently calculate manufacturing statistics or
generate defect predictions.

### Manufacturing Engineer

Retains responsibility for final quality interpretation and further
process investigation.

## Decision Flow

Steel Quality Data

→ Quality Analytics

→ Defect Prediction

→ Prediction Confidence

→ SHAP Model Drivers

→ AI Quality Copilot

→ Engineer Review

→ Further Quality Investigation

## Project Principle

The AI system supports engineering decisions rather than replacing
engineering judgment.
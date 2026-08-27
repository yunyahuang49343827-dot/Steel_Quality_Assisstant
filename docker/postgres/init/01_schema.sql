--
-- PostgreSQL database dump
--

\restrict ZmEZxn9FNFgY02hAAD2cjk4clXqrcigFQNteVoQiEYrXm8t93FiD2jNbFue2DOM

-- Dumped from database version 17.11 (Homebrew)
-- Dumped by pg_dump version 17.11 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: modeling_steel_quality; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.modeling_steel_quality (
    id bigint,
    "X_Minimum" bigint,
    "X_Maximum" bigint,
    "Y_Minimum" bigint,
    "Y_Maximum" bigint,
    "Pixels_Areas" bigint,
    "X_Perimeter" bigint,
    "Y_Perimeter" bigint,
    "Sum_of_Luminosity" bigint,
    "Minimum_of_Luminosity" bigint,
    "Maximum_of_Luminosity" bigint,
    "Length_of_Conveyer" bigint,
    "TypeOfSteel_A300" bigint,
    "TypeOfSteel_A400" bigint,
    "Steel_Plate_Thickness" bigint,
    "Edges_Index" double precision,
    "Empty_Index" double precision,
    "Square_Index" double precision,
    "Outside_X_Index" double precision,
    "Edges_X_Index" double precision,
    "Edges_Y_Index" double precision,
    "Outside_Global_Index" double precision,
    "LogOfAreas" double precision,
    "Log_X_Index" double precision,
    "Log_Y_Index" double precision,
    "Orientation_Index" double precision,
    "Luminosity_Index" double precision,
    "SigmoidOfAreas" double precision,
    "Pastry" bigint,
    "Z_Scratch" bigint,
    "K_Scatch" bigint,
    "Stains" bigint,
    "Dirtiness" bigint,
    "Bumps" bigint,
    "Other_Faults" bigint,
    defect_type text
);


--
-- Name: raw_steel_quality; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.raw_steel_quality (
    id bigint,
    "X_Minimum" bigint,
    "X_Maximum" bigint,
    "Y_Minimum" bigint,
    "Y_Maximum" bigint,
    "Pixels_Areas" bigint,
    "X_Perimeter" bigint,
    "Y_Perimeter" bigint,
    "Sum_of_Luminosity" bigint,
    "Minimum_of_Luminosity" bigint,
    "Maximum_of_Luminosity" bigint,
    "Length_of_Conveyer" bigint,
    "TypeOfSteel_A300" bigint,
    "TypeOfSteel_A400" bigint,
    "Steel_Plate_Thickness" bigint,
    "Edges_Index" double precision,
    "Empty_Index" double precision,
    "Square_Index" double precision,
    "Outside_X_Index" double precision,
    "Edges_X_Index" double precision,
    "Edges_Y_Index" double precision,
    "Outside_Global_Index" double precision,
    "LogOfAreas" double precision,
    "Log_X_Index" double precision,
    "Log_Y_Index" double precision,
    "Orientation_Index" double precision,
    "Luminosity_Index" double precision,
    "SigmoidOfAreas" double precision,
    "Pastry" bigint,
    "Z_Scratch" bigint,
    "K_Scatch" bigint,
    "Stains" bigint,
    "Dirtiness" bigint,
    "Bumps" bigint,
    "Other_Faults" bigint
);


--
-- Name: idx_modeling_defect_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_modeling_defect_type ON public.modeling_steel_quality USING btree (defect_type);


--
-- Name: idx_modeling_steel_quality_id; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_modeling_steel_quality_id ON public.modeling_steel_quality USING btree (id);


--
-- Name: idx_raw_steel_quality_id; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_raw_steel_quality_id ON public.raw_steel_quality USING btree (id);


--
-- PostgreSQL database dump complete
--

\unrestrict ZmEZxn9FNFgY02hAAD2cjk4clXqrcigFQNteVoQiEYrXm8t93FiD2jNbFue2DOM


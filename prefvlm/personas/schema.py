"""Pydantic models for persona representation."""

from typing import Literal

from pydantic import BaseModel, Field


class BigFive(BaseModel):
    openness: Literal["low", "moderate", "high"]
    conscientiousness: Literal["low", "moderate", "high"]
    extraversion: Literal["low", "moderate", "high"]
    agreeableness: Literal["low", "moderate", "high"]
    neuroticism: Literal["low", "moderate", "high"]


class Persona(BaseModel):
    persona_id: str = Field(..., description="Unique identifier, e.g. p001")
    name: str
    age: int = Field(..., ge=18, le=90)
    location: str
    occupation: str
    education_level: str
    data_literacy: Literal["low", "moderate", "high"]
    domain_familiarity: list[str] = Field(..., min_length=1)
    big_five: BigFive
    hobbies: list[str]
    backstory: str = Field(..., description="2-3 sentences about this person's background")

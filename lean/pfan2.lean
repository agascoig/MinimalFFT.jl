-- pfam2.lean - n mapping for two factor prime factor algorithm

-- checked with Lean v4.20.1 June 5, 2025

import Mathlib.Tactic
import Mathlib.Data.Nat.ModEq
import Mathlib.Data.ZMod.Basic
import Mathlib.Algebra.Ring.Basic

/-
  This is not the usual Good-CRT mapping to avoid mod operations
  or look-up table.

  See "A Generalized Mixed-Radix Algorithm for Memory-Based
  FFT Processors."  IEEE Tran. on Circuits and Systems-II.
  Jan 2010.

  See also "Agile Design of Generator-Based Signal Processing Hardware."
  Wang, Angie. UC Berkeley Ph.D. thesis, 2018.

  Given:
  - M₁N₁ + M₂N₂ = 1 (Bezout's identity)
  - N = N₁N₂
  - n = (N₂n₁ + A₁n₂) mod N, where:
    - A₁ = p₁N₁ = q₁N₂ + 1
  - Q₁ = -M₂
  - Q'₁ = (N₁ - Q₁) % N₁

  Need to prove:
  - n = N₂n'₁ + n'₂ = (N₂n₁ + (Q₁N₂ + 1)n₂) mod N

  Candidate solution:
  - n₁ = (n'₁ + Q'₁n'₂) % N₁
  - Q'₁ = (N₁ - Q₁) % N₁
  - n₂ = n'₂
-/

open Nat

-- set_option pp.all true

variable (N N₁ N₂ : ℕ)

lemma I0 (a N : ℕ) (ha : a < N) : (a % N) = a := by
  rw [Nat.mod_eq_of_lt ha]
lemma I1 (a b N : ℕ) : (a*b) % N = (a % N) * (b % N) % N :=
  Nat.mul_mod _ _ _
lemma I2 (a N : ℕ) : (a % N) % N = a % N := Nat.mod_mod _ _
lemma I3 (a b N : ℕ) : (a + b) % N = ((a % N) + (b % N)) % N := Nat.add_mod _ _ _
lemma I4 (a b N : ℕ) :
  ((a % N) + b) % N = (a + b) % N := by
    rw [add_mod] -- (a % N % N + b % N) % N = (a + b) % N
    rw [mod_mod] -- (a % N + b % N) % N = (a + b) % N
    rw [←I3] -- (a % N + b % N) % N = (a + b) % N
lemma I5 (A b N : ℕ) : (A * b) % (A * N) = A * (b % N) := Nat.mul_mod_mul_left _ _ _

lemma L1 (a b c d : ℕ) : (a + (b - c) % b * d + c * d) % b = (a + ((b - c) % b + c) * d) % b := by
  congr 1 -- remove the % b
  ring

lemma L2 (a N : ℕ) (hb : a > 0) (hN : N > 0): (N - a) % N = (N - a) := by
  -- If a < N, then N - a is less than N
  have h2 : N - a < N := by
    apply Nat.sub_lt -- 0 < N
    exact hN -- 0 < a
    exact hb
  rw [Nat.mod_eq_of_lt h2]

theorem pfan2_proof
  (N N₁ N₂ Q₁ : ℕ)
  (h_positive : 0 < N ∧ 0 < N₁ ∧ 0 < N₂)
  (h_Nc : N = N₂ * N₁)
  (h_Q₁_range : 0 < Q₁ ∧ Q₁ < N₁)
  (h_Q'₁ : Q'₁ = (N₁ - Q₁) % N₁)
  (h_n'₁_range : 0 < n'₁ ∧ n'₁ < N₁)
  (h_n₂ : n₂ = n'₂)
  (h_n₁ : n₁ = (n'₁ + Q'₁ * n'₂) % N₁)
  (h_n : (n₂ + N₂ * n'₁) < N):
  (N₂ * n'₁ + n'₂) = (N₂ * n₁ + (Q₁ * N₂ + 1) * n₂) % N := by

  rw [add_mul (Q₁ * N₂) 1 n₂] -- = (N₂ * n₁ + (Q₁ * N₂ * n₂ + 1 * n₂)) % N
  simp
  rw [←add_assoc] -- = (N₂ * n₁ + Q₁ * N₂ * n₂ + n₂) % N
  rw [←mul_comm N₂ Q₁] -- = (N₂ * n₁ + N₂ * Q₁ * n₂ + n₂) % N
  rw [mul_assoc N₂ Q₁ n₂] -- = (N₂ * n₁ + N₂ * (Q₁ * n₂) + n₂) % N
  rw [←mul_add N₂ n₁ (Q₁ * n₂)] -- = (N₂ * (n₁ + Q₁ * n₂) + n₂) % N

  rw [I3 (N₂ * (n₁ + Q₁ * n₂)) n₂ N] -- = (N₂ * (n₁ + Q₁ * n₂) % N + n₂ % N) % N
  rw [h_Nc] -- = (N₂ * (n₁ + Q₁ * n₂) % (N₂ * N₁) + n₂ % (N₂ * N₁)) % (N₂ * N₁)

  rw [I5 N₂ (n₁ + Q₁ * n₂) N₁] -- = (N₂ * ((n₁ + Q₁ * n₂) % N₁) + n₂ % (N₂ * N₁)) % (N₂ * N₁)
  rw [←h_Nc] -- = (N₂ * ((n₁ + Q₁ * n₂) % N₁) + n₂ % N) % N
  conv =>
    rhs -- (N₂ * ((n₁ + Q₁ * n₂) % N₁) + n₂ % N) % N
    rw [add_comm] -- (n₂ % N + N₂ * ((n₁ + Q₁ * n₂) % N₁)) % N

  rw [I4 n₂ (N₂ * ((n₁ + Q₁ * n₂) % N₁)) N] -- = (n₂ + N₂ * ((n₁ + Q₁ * n₂) % N₁)) % N

  -- Need to show the inner modulus reduces to n'₁.

  have e1 : ((n₁ + Q₁ * n₂) % N₁) = n'₁ := by
    rw [h_n₁] -- ((n'₁ + Q'₁ * n'₂) % N₁ + Q₁ * n₂) % N₁ = n'₁
    rw [←h_n₂] -- ((n'₁ + Q'₁ * n₂) % N₁ + Q₁ * n₂) % N₁ = n'₁
    rw [h_Q'₁] -- ((n'₁ + (N₁ - Q₁) % N₁ * n₂) % N₁ + Q₁ * n₂) % N₁ = n'₁
    rw [I4 (n'₁ + (N₁ - Q₁) % N₁ * n₂) (Q₁ * n₂) N₁] -- (n'₁ + (N₁ - Q₁) % N₁ * n₂ + Q₁ * n₂) % N₁ = n'₁
    rw [L1] -- (n'₁ + ((N₁ - Q₁) % N₁ + Q₁) * n₂) % N₁ = n'₁
    rw [I3 n'₁ (((N₁ - Q₁) % N₁ + Q₁) * n₂) N₁] -- (n'₁ % N₁ + ((N₁ - Q₁) % N₁ + Q₁) * n₂ % N₁) % N₁ = n'₁
    rw [I1] -- (n'₁ % N₁ + ((N₁ - Q₁) % N₁ + Q₁) % N₁ * (n₂ % N₁) % N₁) % N₁ = n'₁

    have h_n'₁ : n'₁ % N₁ = n'₁ := by
      rw [I0 n'₁ N₁ h_n'₁_range.2]

    have h_Q₁_pos : Q₁ > 0 := h_Q₁_range.1
    have h_N₁_pos : N₁ > 0 := h_positive.2.1
    rw [L2 Q₁ N₁ h_Q₁_pos h_N₁_pos] -- (n'₁ % N₁ + (N₁ - Q₁ + Q₁) % N₁ * (n₂ % N₁) % N₁) % N₁ = n'₁

    have h_Q₁_le : Q₁ ≤ N₁ := Nat.le_of_lt h_Q₁_range.2
    rw [Nat.sub_add_cancel h_Q₁_le] -- (n'₁ % N₁ + N₁ % N₁ * (n₂ % N₁) % N₁) % N₁ = n'₁
    simp only [Nat.mod_self] -- (n'₁ % N₁ + 0 * (n₂ % N₁) % N₁) % N₁ = n'₁
    simp -- n'₁ % N₁ = n'₁
    rw [h_n'₁]

  rw [e1] -- N₂ * n'₁ + n'₂ = (n₂ + N₂ * n'₁) % N
  rw [I0 (n₂ + N₂ * n'₁) N h_n] -- N₂ * n'₁ + n'₂ = n₂ + N₂ * n'₁
  rw [add_comm] -- n'₂ + N₂ * n'₁ = n₂ + N₂ * n'₁
  rw [h_n₂]

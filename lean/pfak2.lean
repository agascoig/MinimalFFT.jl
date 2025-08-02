
import Mathlib.Tactic
import Mathlib.Data.Nat.ModEq
import Mathlib.Data.ZMod.Basic
import Mathlib.Algebra.Ring.Basic
import Mathlib.Data.Int.Basic

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
  - k = (B₁k₁ + N₁k₂) mod N, where:
    - B₁ = p₂N₂ = q₂N₁ + 1
    - B₁ = N₂ × (N₂⁻¹ mod N₁)
    - k_forward = (N₂ × N₂⁻¹ × k₁ + N₁k₂) mod N
  - Q₁ = -M₂ (thesis 2.77)
  - Q₂ = -M₁ = q₂
  - Q'₂ = (N₂ - Q₂) mod N₂

  Solution for k:
  - k'₁ ∈ {1,...,N₁-1}
  - k'₂ ∈ {1,...,N₂-1}
  - k₁ = k'₁
  - k₂ = (k'₂ + Q'₂ × k'₁) mod N₂

  We need to prove that with these k₁, k₂:
  (k'₁ + N₁k'₂) = ((Q₂N₁ + 1)k₁ + N₁k₂) mod N

  Also:
  - Q'₁ = (N₁ - Q₁) mod N₁
  - The relationship Q'₁ + Q₁ ≡ 0 (mod N₁)

  Candidate solution:
  - k₁ = k'₁
  - Q'₂ = (N₂ - Q₂) mod N₂
  - k₂ = (k'₂ + Q'₂ × k'₁) % N₂
-/

open Int

variable (N N₁ N₂ : ℤ)

lemma I0 (a N : ℤ) (ha : 0 ≤ a ∧ a < N) : (a % N) = a := by
  exact Int.emod_eq_of_lt ha.1 ha.2
lemma I1 (a b N : ℤ) : (a*b) % N = (a % N) * (b % N) % N :=
  Int.mul_emod _ _ _
lemma I2 (a N : ℤ) : (a % N) % N = a % N := Int.emod_emod _ _
lemma I3 (a b N : ℤ) : (a + b) % N = ((a % N) + (b % N)) % N := Int.add_emod _ _ _
lemma I4 (a b N : ℤ) :
  ((a % N) + b) % N = (a + b) % N := by
  rw [add_emod] -- (a % N % N + b % N) % N = (a + b) % N
  rw [emod_emod] -- (a % N + b % N) % N = (a + b) % N
  rw [←I3] -- (a % N + b % N) % N = (a + b) % N
lemma I5 (A b N : ℤ) (hA: 0 < A) : (A * b) % (A * N) = A * (b % N) := by
  rw [mul_emod_mul_of_pos _ _ hA]

lemma L1 (a N : ℤ) (ha : 0 < a ∧ a < N) : (N - a) % N = (N - a) := by
  apply Int.emod_eq_of_lt
  · linarith [ha.2]  -- 0 ≤ N - a from a < N
  · linarith [ha.1]  -- N - a < N from 0 < a

lemma L2 (a b c : ℤ) : a + (b - c) = a + b - c := by
  rw [add_comm]
  ring

theorem pfak2_proof
  (N N₁ N₂ Q₂ : ℤ)
  (h_positive : 0 < N ∧ 0 < N₁ ∧ 0 < N₂) -- asserts these are positive integers
  (h_N : N = N₁ * N₂)
--  (h_Nc : N = N₂ * N₁)
  (h_Q₂_range : 0 < Q₂ ∧ Q₂ < N₂)
  (h_Q'₂ : Q'₂ = (N₂ - Q₂) % N₂)
--  (h_k'₁_range : 0 < k'₁ ∧ k'₁ < N₁)
--  (h_k'₂_range : 0 < k'₂ ∧ k'₂ < N₂)
  (h_k₁ : k₁ = k'₁)
  (h_k₂ : k₂ = (k'₂ + Q'₂ * k'₁) % N₂)
  (h_k : 0 ≤ (N₁ * k'₂ + k'₁) ∧ (N₁ * k'₂ + k'₁) < N) :
  (k'₁ + N₁ * k'₂) = ((Q₂ * N₁ + 1) * k₁ + N₁ * k₂) % N := by

  rw [h_k₁] -- = ((Q₂ + 1) * k'₁ + N₁ * k₂) % N
  rw [h_k₂] -- = ((Q₂ + 1) * k'₁ + N₁ * ((k'₂ + Q'₂ * k'₁) % N₂)) % N
  rw [h_Q'₂] -- = ((Q₂ + 1) * k'₁ + N₁ * ((k'₂ + (N₂ - Q₂) % N₂ * k'₁) % N₂)) % N

  rw [L1 Q₂ N₂ h_Q₂_range] -- ((Q₂ * N₁ + 1) * k'₁ + N₁ * ((k'₂ + (N₂ - Q₂) * k'₁) % N₂)) % N
  rw [sub_mul] -- ((Q₂ * N₁ + 1) * k'₁ + N₁ * ((k'₂ + (N₂ * k'₁ - Q₂ * k'₁)) % N₂)) % N
  rw [L2] -- ((Q₂ * N₁ + 1) * k'₁ + N₁ * ((k'₂ + N₂ * k'₁ - Q₂ * k'₁) % N₂)) % N
  have e1 (a b c : ℤ): a + b - c = a - c + b := by
    ring
  rw [e1] -- ((Q₂ * N₁ + 1) * k'₁ + N₁ * ((k'₂ - Q₂ * k'₁ + N₂ * k'₁) % N₂)) % N
  rw [add_mul_emod_self_left] -- ((Q₂ * N₁ + 1) * k'₁ + N₁ * ((k'₂ - Q₂ * k'₁) % N₂)) % N
  rw [←I5 N₁ (k'₂ - Q₂ * k'₁) N₂ h_positive.2.1] -- ((Q₂ * N₁ + 1) * k'₁ + N₁ * (k'₂ - Q₂ * k'₁) % (N₁ * N₂)) % N
  rw [←h_N] -- ((Q₂ * N₁ + 1) * k'₁ + N₁ * (k'₂ - Q₂ * k'₁) % N) % N

  have e2 (a b N : ℤ) : a + b % N = b % N + a := by
    ring
  rw [e2] -- (N₁ * (k'₂ - Q₂ * k'₁) % N + (Q₂ * N₁ + 1) * k'₁) % N
  rw [I4] -- (N₁ * (k'₂ - Q₂ * k'₁) + (Q₂ * N₁ + 1) * k'₁) % N
  rw [mul_sub] -- (N₁ * k'₂ - N₁ * (Q₂ * k'₁) + (Q₂ * N₁ + 1) * k'₁) % N
  rw [add_mul] -- (N₁ * k'₂ - N₁ * (Q₂ * k'₁) + (Q₂ * N₁ * k'₁ + 1 * k'₁)) % N
  rw [one_mul] -- (N₁ * k'₂ - N₁ * (Q₂ * k'₁) + (Q₂ * N₁ * k'₁ + k'₁)) % N
  rw [←e1] -- (N₁ * k'₂ + (Q₂ * N₁ * k'₁ + k'₁) - N₁ * (Q₂ * k'₁)) % N
  have reorder1 : Q₂ * N₁ * k'₁ = N₁ * Q₂ * k'₁ := by
    ring
  rw [reorder1] -- (N₁ * k'₂ + (N₁ * Q₂ * k'₁ + k'₁) - N₁ * (Q₂ * k'₁)) % N
  have reorder2 : (N₁ * k'₂ + (N₁ * Q₂ * k'₁ + k'₁) - N₁ * (Q₂ * k'₁)) = N₁ * k'₂ + k'₁ + N₁ * Q₂ * k'₁ - N₁ * Q₂ * k'₁ := by
    ring
  rw [reorder2] -- (N₁ * k'₂ + k'₁ + N₁ * Q₂ * k'₁ - N₁ * Q₂ * k'₁) % N
  rw [add_sub_cancel_right (N₁ * k'₂ + k'₁) (N₁ * Q₂ * k'₁)] -- (N₁ * k'₂ + k'₁) % N
  rw [I0 (N₁ * k'₂ + k'₁) N h_k] -- N₁ * k'₂ + k'₁
  rw [add_comm] -- Goals accomplished!

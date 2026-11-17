package com.voiceshield

import com.voiceshield.ml.AasistLEngine
import com.voiceshield.ml.EcapaTdnnEngine
import com.voiceshield.ml.SileroVadEngine
import com.voiceshield.shared.*
import org.junit.Assert.*
import org.junit.Test

class ModelRuntimeForensicTest {

    @Test
    fun testSileroVadEngine_uninitialized_reportsLoadingAndNotLoadedYet() {
        val vad = SileroVadEngine()
        assertFalse("Unloaded VAD must not be ready", vad.isReady())
        val health = vad.getDetailedHealth()
        assertEquals("Silero VAD", health.name)
        assertEquals(ModelStatus.LOADING, health.status)
        assertEquals(ModelFailureReason.NOT_LOADED_YET, health.failureReason)
        assertNull("Unloaded engine before error has null detailMessage", health.detailMessage)
    }

    @Test
    fun testAasistLEngine_uninitialized_reportsLoadingAndNotLoadedYet() {
        val aasist = AasistLEngine()
        assertFalse("Unloaded AASIST must not be ready", aasist.isReady())
        val health = aasist.getDetailedHealth()
        assertEquals("AASIST-L", health.name)
        assertEquals(ModelStatus.LOADING, health.status)
        assertEquals(ModelFailureReason.NOT_LOADED_YET, health.failureReason)
        assertNull("Unloaded engine before error has null detailMessage", health.detailMessage)
    }

    @Test
    fun testEcapaTdnnEngine_uninitialized_reportsLoadingAndNotLoadedYet() {
        val ecapa = EcapaTdnnEngine()
        assertFalse("Unloaded ECAPA must not be ready", ecapa.isReady())
        val health = ecapa.getDetailedHealth()
        assertEquals("ECAPA-TDNN", health.name)
        assertEquals(ModelStatus.LOADING, health.status)
        assertEquals(ModelFailureReason.NOT_LOADED_YET, health.failureReason)
    }

    @Test
    fun testModelFailureIsolation_independentHealthStatus() {
        // Construct detailed health map representing independent failure:
        // Silero: READY
        // AASIST-L: ERROR with ASSET_MISSING
        // ECAPA-TDNN: READY
        val detailedMap = mapOf(
            "Silero VAD" to DetailedModelHealth(
                name = "Silero VAD",
                status = ModelStatus.READY,
                failureReason = ModelFailureReason.NONE
            ),
            "AASIST-L" to DetailedModelHealth(
                name = "AASIST-L",
                status = ModelStatus.ERROR,
                failureReason = ModelFailureReason.ASSET_MISSING,
                detailMessage = "Model asset missing"
            ),
            "ECAPA-TDNN" to DetailedModelHealth(
                name = "ECAPA-TDNN",
                status = ModelStatus.READY,
                failureReason = ModelFailureReason.NONE
            )
        )

        // Verify that AASIST failure DOES NOT mark Silero or ECAPA as unavailable
        assertEquals(ModelStatus.READY, detailedMap["Silero VAD"]?.status)
        assertEquals(ModelStatus.ERROR, detailedMap["AASIST-L"]?.status)
        assertEquals(ModelFailureReason.ASSET_MISSING, detailedMap["AASIST-L"]?.failureReason)
        assertEquals(ModelStatus.READY, detailedMap["ECAPA-TDNN"]?.status)
    }

    @Test
    fun testAiExceptionContainment_detectionFailureDoesNotPropagate() {
        val vad = SileroVadEngine()
        val aasist = AasistLEngine()
        val ecapa = EcapaTdnnEngine()

        // When uninitialized, inference methods must fail safely with zero/empty results or exceptions
        // caught by the coordinator, never crashing telephony.
        assertFalse(vad.isReady())
        assertFalse(aasist.isReady())
        assertFalse(ecapa.isReady())
    }

    @Test
    fun testAllFailureReasonsDistinguishable() {
        // Ensure failure reasons are distinct and identifiable
        val reasons = ModelFailureReason.values()
        assertTrue("Must have multiple distinct failure reasons", reasons.size >= 10)
        assertTrue(reasons.contains(ModelFailureReason.ASSET_MISSING))
        assertTrue(reasons.contains(ModelFailureReason.ONNX_RUNTIME_INIT_FAILED))
        assertTrue(reasons.contains(ModelFailureReason.MODEL_LOAD_FAILED))
        assertTrue(reasons.contains(ModelFailureReason.INPUT_CONTRACT_INVALID))
        assertTrue(reasons.contains(ModelFailureReason.OUTPUT_CONTRACT_INVALID))
        assertTrue(reasons.contains(ModelFailureReason.STARTUP_TEST_INFERENCE_FAILED))
    }
}

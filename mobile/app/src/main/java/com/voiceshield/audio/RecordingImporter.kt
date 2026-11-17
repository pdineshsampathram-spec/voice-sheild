/**
 * RecordingImporter — Handles recording selection via Android Storage Access Framework.
 *
 * Uses SAF (ActivityResultContracts.OpenDocument) exclusively.
 * No OEM-specific path scanning, no root, no hidden APIs.
 * Recording is user-owned — never modified, deleted, or copied.
 *
 * All I/O via ContentResolver — no filesystem path conversion.
 */
package com.voiceshield.audio

import android.content.ContentResolver
import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log

/**
 * Result of importing a recording via SAF.
 */
data class RecordingImportResult(
    val uri: Uri,
    val displayName: String,
    val sizeBytes: Long,
    val mimeType: String?
)

/**
 * Helpers for recording import via SAF.
 * The actual ActivityResultLauncher is registered in MainActivity.
 */
object RecordingImporter {

    private const val TAG = "RecordingImporter"

    /** MIME types accepted by the SAF file picker. */
    val ACCEPTED_MIME_TYPES = arrayOf(
        "audio/mpeg",       // MP3
        "audio/mp4",        // M4A/AAC
        "audio/x-m4a",      // M4A alternate
        "audio/wav",        // WAV
        "audio/x-wav",      // WAV alternate
        "audio/3gpp",       // 3GP
        "audio/amr",        // AMR
        "audio/amr-wb",     // AMR-WB
        "audio/ogg",        // OGG
        "audio/flac",       // FLAC
        "audio/*"           // Fallback: any audio
    )

    /**
     * Query metadata for a selected recording Uri.
     * Returns null if the Uri is invalid or inaccessible.
     */
    fun queryRecordingMetadata(context: Context, uri: Uri): RecordingImportResult? {
        return try {
            val contentResolver = context.contentResolver
            var displayName = "Unknown"
            var sizeBytes = 0L

            contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    val sizeIndex = cursor.getColumnIndex(OpenableColumns.SIZE)
                    if (nameIndex >= 0) {
                        displayName = cursor.getString(nameIndex) ?: "Unknown"
                    }
                    if (sizeIndex >= 0) {
                        sizeBytes = cursor.getLong(sizeIndex)
                    }
                }
            }

            val mimeType = contentResolver.getType(uri)

            // Validate: file must be non-empty
            if (sizeBytes <= 0) {
                Log.w(TAG, "Recording file is empty: $displayName")
                return null
            }

            Log.i(TAG, "Recording selected: $displayName ($sizeBytes bytes, $mimeType)")

            RecordingImportResult(
                uri = uri,
                displayName = displayName,
                sizeBytes = sizeBytes,
                mimeType = mimeType
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to query recording metadata: ${e.message}", e)
            null
        }
    }

    /**
     * Verify that the ContentResolver can open the Uri for reading.
     */
    fun canRead(context: Context, uri: Uri): Boolean {
        return try {
            context.contentResolver.openInputStream(uri)?.use { true } ?: false
        } catch (e: Exception) {
            Log.w(TAG, "Cannot read recording: ${e.message}")
            false
        }
    }
}

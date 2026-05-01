/** Read image file as data URL (backend accepts raw base64 or full data:image/... URL). */
export function fileToDataUrl(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => resolve(reader.result as string)
        reader.onerror = () => reject(reader.error ?? new Error('read failed'))
        reader.readAsDataURL(file)
    })
}

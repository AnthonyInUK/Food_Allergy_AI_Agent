import axios, { AxiosInstance } from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const apiClient: AxiosInstance = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
})

export type ReasoningStep = {
    key: string
    label: string
    seconds?: number
    detail?: string
}

export type MessageMeta = {
    reasoning_trace?: ReasoningStep[]
    total_seconds?: number
    cached?: boolean
    /** 流式生成中：为 true 时展开思考过程并显示进行中 */
    streaming?: boolean
    partial_total_seconds?: number
}

export interface Message {
    text: string
    role: 'user' | 'assistant'
    meta?: MessageMeta
}

export interface Conversation {
    id: string
    title: string
    messages: Message[]
    created_at?: string
    last_reply_cached?: boolean
}

export interface ChatResponse {
    response: string
    conversation_id: string
    cached?: boolean
    reasoning_trace?: ReasoningStep[]
    total_seconds?: number
}

export type ChatStreamEvent =
    | { type: 'start'; conversation_id: string }
    | { type: 'step'; reasoning_trace: ReasoningStep[]; partial_total_seconds?: number }
    /** 答案正文增量（LLM token 流） */
    | { type: 'delta'; text: string }
    | {
          type: 'done'
          response: string
          reasoning_trace: ReasoningStep[]
          total_seconds: number
          cached: boolean
          conversation_id: string
      }
    | { type: 'error'; message: string }

async function readNDJSONStream(response: Response, onLine: (obj: Record<string, unknown>) => void): Promise<void> {
    const reader = response.body?.getReader()
    if (!reader) throw new Error('No response body')
    const dec = new TextDecoder()
    let buf = ''
    while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        const parts = buf.split('\n')
        buf = parts.pop() ?? ''
        for (const line of parts) {
            const t = line.trim()
            if (!t) continue
            try {
                onLine(JSON.parse(t) as Record<string, unknown>)
            } catch (e) {
                console.warn('NDJSON parse skipped:', t.slice(0, 120), e)
            }
        }
    }
    const rest = buf.trim()
    if (rest) {
        try {
            onLine(JSON.parse(rest) as Record<string, unknown>)
        } catch (e) {
            console.warn('NDJSON trailing parse failed:', rest.slice(0, 120), e)
        }
    }
}

export const conversationAPI = {
    listAll: async (): Promise<Conversation[]> => {
        const res = await apiClient.get('/api/conversations')
        return res.data.conversations
    },

    getById: async (id: string): Promise<Conversation> => {
        const res = await apiClient.get(`/api/conversations/${id}`)
        return res.data.conversation
    },

    create: async (title?: string): Promise<Conversation> => {
        const res = await apiClient.post('/api/conversations', { title })
        return res.data.conversation
    },

    delete: async (id: string): Promise<void> => {
        await apiClient.delete(`/api/conversations/${id}`)
    },

    bulkDelete: async (ids: string[]): Promise<number> => {
        if (!ids.length) return 0
        const res = await apiClient.post('/api/conversations/bulk-delete', {
            conversation_ids: ids,
        })
        return Number(res.data?.deleted_count ?? 0)
    },

    update: async (id: string, updates: Partial<Conversation>): Promise<Conversation> => {
        const res = await apiClient.put(`/api/conversations/${id}`, updates)
        return res.data.conversation
    },
}

export const chatAPI = {
    sendMessage: async (
        text: string,
        conversationId?: string,
        imageBase64?: string | null
    ): Promise<ChatResponse> => {
        const body: Record<string, unknown> = { text, conversation_id: conversationId }
        if (imageBase64) body.image_base64 = imageBase64
        const res = await apiClient.post('/api/chat', body)
        return res.data
    },

    uploadImage: async (file: File, conversationId?: string): Promise<ChatResponse> => {
        const formData = new FormData()
        formData.append('file', file)
        if (conversationId) {
            formData.append('conversation_id', conversationId)
        }

        const res = await apiClient.post('/api/upload-image', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        })
        return res.data
    },

    sendMessageStream: async (
        text: string,
        conversationId: string | undefined,
        onEvent: (e: ChatStreamEvent) => void,
        imageBase64?: string | null
    ): Promise<void> => {
        const body: Record<string, unknown> = { text, conversation_id: conversationId ?? null }
        if (imageBase64) body.image_base64 = imageBase64
        const res = await fetch(`${API_BASE_URL}/api/chat/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                Accept: 'application/x-ndjson',
            },
            body: JSON.stringify(body),
        })
        if (!res.ok) {
            const t = await res.text()
            throw new Error(t || res.statusText)
        }
        await readNDJSONStream(res, (row) => {
            const typ = row.type as string
            if (typ === 'start') {
                onEvent({ type: 'start', conversation_id: String(row.conversation_id) })
            } else if (typ === 'step') {
                onEvent({
                    type: 'step',
                    reasoning_trace: (row.reasoning_trace as ReasoningStep[]) || [],
                    partial_total_seconds: row.partial_total_seconds as number | undefined,
                })
            } else if (typ === 'delta') {
                onEvent({ type: 'delta', text: String(row.text ?? '') })
            } else if (typ === 'done') {
                onEvent({
                    type: 'done',
                    response: String(row.response ?? ''),
                    reasoning_trace: (row.reasoning_trace as ReasoningStep[]) || [],
                    total_seconds: Number(row.total_seconds ?? 0),
                    cached: Boolean(row.cached),
                    conversation_id: String(row.conversation_id),
                })
            } else if (typ === 'error') {
                onEvent({ type: 'error', message: String(row.message ?? 'Unknown error') })
            }
        })
    },

    uploadImageStream: async (
        file: File,
        conversationId: string | undefined,
        onEvent: (e: ChatStreamEvent) => void
    ): Promise<void> => {
        const fd = new FormData()
        fd.append('file', file)
        if (conversationId) fd.append('conversation_id', conversationId)
        const res = await fetch(`${API_BASE_URL}/api/upload-image/stream`, {
            method: 'POST',
            headers: { Accept: 'application/x-ndjson' },
            body: fd,
        })
        if (!res.ok) {
            const t = await res.text()
            throw new Error(t || res.statusText)
        }
        await readNDJSONStream(res, (row) => {
            const typ = row.type as string
            if (typ === 'start') {
                onEvent({ type: 'start', conversation_id: String(row.conversation_id) })
            } else if (typ === 'step') {
                onEvent({
                    type: 'step',
                    reasoning_trace: (row.reasoning_trace as ReasoningStep[]) || [],
                    partial_total_seconds: row.partial_total_seconds as number | undefined,
                })
            } else if (typ === 'done') {
                onEvent({
                    type: 'done',
                    response: String(row.response ?? ''),
                    reasoning_trace: (row.reasoning_trace as ReasoningStep[]) || [],
                    total_seconds: Number(row.total_seconds ?? 0),
                    cached: Boolean(row.cached),
                    conversation_id: String(row.conversation_id),
                })
            } else if (typ === 'error') {
                onEvent({ type: 'error', message: String(row.message ?? 'Unknown error') })
            }
        })
    },
}

export default apiClient

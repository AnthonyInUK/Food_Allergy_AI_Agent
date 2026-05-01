'use client'

import React, { createContext, useContext, useState, useCallback, useEffect } from 'react'
import { Conversation, Message, ReasoningStep, chatAPI, conversationAPI } from '@/lib/api'
import { fileToDataUrl } from '@/lib/image'

interface ConversationContextType {
    conversations: Conversation[]
    currentConversation: Conversation | null
    isLoading: boolean
    error: string | null

    loadConversations: () => Promise<void>
    selectConversation: (id: string) => Promise<void>
    createConversation: (title?: string) => Promise<Conversation | null>
    deleteConversation: (id: string) => Promise<void>
    deleteConversations: (ids: string[]) => Promise<void>

    sendMessage: (text: string, opts?: { image?: File | null }) => Promise<void>
    uploadImage: (file: File) => Promise<void>
}

const ConversationContext = createContext<ConversationContextType | undefined>(undefined)

export const ConversationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [conversations, setConversations] = useState<Conversation[]>([])
    const [currentConversation, setCurrentConversation] = useState<Conversation | null>(null)
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    const loadConversations = useCallback(async () => {
        setIsLoading(true)
        setError(null)
        try {
            const data = await conversationAPI.listAll()
            setConversations(data)

            if (data.length > 0) {
                setCurrentConversation(data[0])
            }
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Failed to load conversations'
            setError(message)
            console.error('Error loading conversations:', err)
        } finally {
            setIsLoading(false)
        }
    }, [])

    const selectConversation = useCallback(async (id: string) => {
        setIsLoading(true)
        setError(null)
        try {
            const conv = await conversationAPI.getById(id)
            setCurrentConversation(conv)
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Failed to select conversation'
            setError(message)
            console.error('Error selecting conversation:', err)
        } finally {
            setIsLoading(false)
        }
    }, [])

    const createConversation = useCallback(async (title?: string): Promise<Conversation | null> => {
        setIsLoading(true)
        setError(null)
        try {
            const newConv = await conversationAPI.create(title)
            setConversations((prev) => [newConv, ...prev])
            setCurrentConversation(newConv)
            return newConv
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Failed to create conversation'
            setError(message)
            console.error('Error creating conversation:', err)
            return null
        } finally {
            setIsLoading(false)
        }
    }, [])

    const deleteConversation = useCallback(
        async (id: string) => {
            setIsLoading(true)
            setError(null)
            try {
                await conversationAPI.delete(id)
                setConversations((prev) => prev.filter((c) => c.id !== id))

                if (currentConversation?.id === id) {
                    const remaining = conversations.filter((c) => c.id !== id)
                    if (remaining.length > 0) {
                        setCurrentConversation(remaining[0])
                    } else {
                        const newConv = await conversationAPI.create()
                        setConversations([newConv])
                        setCurrentConversation(newConv)
                    }
                }
            } catch (err) {
                const message = err instanceof Error ? err.message : 'Failed to delete conversation'
                setError(message)
                console.error('Error deleting conversation:', err)
            } finally {
                setIsLoading(false)
            }
        },
        [currentConversation, conversations]
    )

    const deleteConversations = useCallback(
        async (ids: string[]) => {
            const uniq = Array.from(new Set(ids.filter(Boolean)))
            if (!uniq.length) return
            setIsLoading(true)
            setError(null)
            try {
                await conversationAPI.bulkDelete(uniq)
                const remaining = conversations.filter((c) => !uniq.includes(c.id))
                setConversations(remaining)
                if (currentConversation && uniq.includes(currentConversation.id)) {
                    if (remaining.length > 0) {
                        setCurrentConversation(remaining[0])
                    } else {
                        const newConv = await conversationAPI.create()
                        setConversations([newConv])
                        setCurrentConversation(newConv)
                    }
                }
            } catch (err) {
                const message = err instanceof Error ? err.message : 'Failed to bulk delete conversations'
                setError(message)
                console.error('Error bulk deleting conversations:', err)
            } finally {
                setIsLoading(false)
            }
        },
        [conversations, currentConversation]
    )

    const sendMessage = useCallback(
        async (text: string, opts?: { image?: File | null }) => {
            const imageFile = opts?.image ?? null
            let imageBase64: string | undefined
            if (imageFile) {
                try {
                    imageBase64 = await fileToDataUrl(imageFile)
                } catch (e) {
                    console.error('Failed to read image:', e)
                    setError('无法读取图片，请重试')
                    return
                }
            }

            const userLine =
                imageFile != null
                    ? `${text}\n📷 [附图: ${imageFile.name}]`
                    : text

            let conv = currentConversation
            if (!conv) {
                const seed = text.trim().slice(0, 28) || '📷 附图提问'
                conv = await createConversation(seed + (seed.length >= 28 ? '…' : ''))
            }
            if (!conv) return

            setError(null)
            try {
                const updatedConv = {
                    ...conv,
                    messages: [...conv.messages, { text: userLine, role: 'user' as const }],
                }

                const cid = conv.id
                const placeholder = {
                    text: '',
                    role: 'assistant' as const,
                    meta: {
                        reasoning_trace: [] as ReasoningStep[],
                        streaming: true as const,
                    },
                }
                const withAssistant: Conversation = {
                    ...updatedConv,
                    messages: [...updatedConv.messages, placeholder],
                }
                setCurrentConversation(withAssistant)
                setConversations((prev) =>
                    prev.map((c) => (c.id === cid ? { ...c, messages: withAssistant.messages } : c))
                )

                type StreamEnd = 'none' | 'done' | 'error'
                let streamEnd: StreamEnd = 'none'
                await chatAPI.sendMessageStream(text, cid, (ev) => {
                    if (ev.type === 'delta') {
                        const piece = ev.text
                        if (!piece) return
                        setCurrentConversation((prev) => {
                            if (!prev || prev.id !== cid) return prev
                            const msgs = [...prev.messages]
                            const li = msgs.length - 1
                            const last = msgs[li]
                            if (!last || last.role !== 'assistant') return prev
                            msgs[li] = {
                                ...last,
                                text: (last.text || '') + piece,
                                meta: {
                                    ...last.meta,
                                    streaming: true,
                                },
                            }
                            return { ...prev, messages: msgs }
                        })
                        setConversations((prev) =>
                            prev.map((c) => {
                                if (c.id !== cid) return c
                                const msgs = [...c.messages]
                                const li = msgs.length - 1
                                const last = msgs[li]
                                if (!last || last.role !== 'assistant') return c
                                msgs[li] = {
                                    ...last,
                                    text: (last.text || '') + piece,
                                    meta: {
                                        ...last.meta,
                                        streaming: true,
                                    },
                                }
                                return { ...c, messages: msgs }
                            })
                        )
                    } else if (ev.type === 'step') {
                        setCurrentConversation((prev) => {
                            if (!prev || prev.id !== cid) return prev
                            const msgs = [...prev.messages]
                            const li = msgs.length - 1
                            const last = msgs[li]
                            if (!last || last.role !== 'assistant') return prev
                            msgs[li] = {
                                ...last,
                                meta: {
                                    ...last.meta,
                                    reasoning_trace: ev.reasoning_trace,
                                    streaming: true,
                                    partial_total_seconds: ev.partial_total_seconds,
                                },
                            }
                            return { ...prev, messages: msgs }
                        })
                    } else if (ev.type === 'done') {
                        streamEnd = 'done'
                        setCurrentConversation((prev) => {
                            if (!prev || prev.id !== cid) return prev
                            const msgs = [...prev.messages]
                            const li = msgs.length - 1
                            if (li < 0 || msgs[li].role !== 'assistant') return prev
                            msgs[li] = {
                                text: ev.response,
                                role: 'assistant',
                                meta: {
                                    reasoning_trace: ev.reasoning_trace,
                                    total_seconds: ev.total_seconds,
                                    cached: ev.cached,
                                    streaming: false,
                                },
                            }
                            return { ...prev, messages: msgs, id: ev.conversation_id }
                        })
                        setConversations((p) =>
                            p.map((c) => {
                                if (c.id !== cid) return c
                                const msgs = [...c.messages]
                                const li = msgs.length - 1
                                if (li < 0 || msgs[li].role !== 'assistant') return c
                                msgs[li] = {
                                    text: ev.response,
                                    role: 'assistant',
                                    meta: {
                                        reasoning_trace: ev.reasoning_trace,
                                        total_seconds: ev.total_seconds,
                                        cached: ev.cached,
                                        streaming: false,
                                    },
                                }
                                return { ...c, messages: msgs, id: ev.conversation_id, last_reply_cached: ev.cached }
                            })
                        )
                    } else if (ev.type === 'error') {
                        streamEnd = 'error'
                        setError(ev.message)
                        setCurrentConversation((prev) => {
                            if (!prev) return prev
                            return { ...prev, messages: prev.messages.slice(0, -1) }
                        })
                    }
                }, imageBase64)

                if (streamEnd === 'none') {
                    const res = await chatAPI.sendMessage(text, cid, imageBase64)
                    setCurrentConversation((prev) => {
                        if (!prev || prev.id !== cid) return prev
                        const msgs = [...prev.messages]
                        const li = msgs.length - 1
                        if (li < 0 || msgs[li].role !== 'assistant') return prev
                        msgs[li] = {
                            text: res.response,
                            role: 'assistant',
                            meta: {
                                reasoning_trace: res.reasoning_trace ?? [],
                                total_seconds: res.total_seconds,
                                cached: res.cached,
                                streaming: false,
                            },
                        }
                        return { ...prev, messages: msgs, id: res.conversation_id }
                    })
                    setConversations((p) =>
                        p.map((c) => {
                            if (c.id !== cid) return c
                            const msgs = [...c.messages]
                            const li = msgs.length - 1
                            if (li < 0 || msgs[li].role !== 'assistant') return c
                            msgs[li] = {
                                text: res.response,
                                role: 'assistant',
                                meta: {
                                    reasoning_trace: res.reasoning_trace ?? [],
                                    total_seconds: res.total_seconds,
                                    cached: res.cached,
                                    streaming: false,
                                },
                            }
                            return { ...c, messages: msgs, id: res.conversation_id, last_reply_cached: !!res.cached }
                        })
                    )
                }
            } catch (err) {
                const message = err instanceof Error ? err.message : 'Failed to send message'
                setError(message)
                console.error('Error sending message:', err)
                try {
                    const res = await chatAPI.sendMessage(text, conv.id, imageBase64)
                    setCurrentConversation((prev) => {
                        if (!prev || prev.id !== conv.id) return prev
                        const msgs = [...prev.messages]
                        const li = msgs.length - 1
                        if (li >= 0 && msgs[li].role === 'assistant' && !msgs[li].text) {
                            msgs[li] = {
                                text: res.response,
                                role: 'assistant',
                                meta: {
                                    reasoning_trace: res.reasoning_trace ?? [],
                                    total_seconds: res.total_seconds,
                                    cached: res.cached,
                                    streaming: false,
                                },
                            }
                        }
                        return { ...prev, messages: msgs, id: res.conversation_id }
                    })
                    setConversations((p) =>
                        p.map((c) => {
                            if (c.id !== conv.id) return c
                            const msgs = [...c.messages]
                            const li = msgs.length - 1
                            if (li >= 0 && msgs[li].role === 'assistant' && !msgs[li].text) {
                                msgs[li] = {
                                    text: res.response,
                                    role: 'assistant',
                                    meta: {
                                        reasoning_trace: res.reasoning_trace ?? [],
                                        total_seconds: res.total_seconds,
                                        cached: res.cached,
                                        streaming: false,
                                    },
                                }
                            }
                            return { ...c, messages: msgs, id: res.conversation_id, last_reply_cached: !!res.cached }
                        })
                    )
                    setError(null)
                } catch {
                    setCurrentConversation((prev) => {
                        if (!prev) return prev
                        const msgs = [...prev.messages]
                        const last = msgs[msgs.length - 1]
                        if (last?.role === 'assistant' && !last.text) msgs.pop()
                        return { ...prev, messages: msgs }
                    })
                }
            }
        },
        [currentConversation, createConversation]
    )

    const uploadImage = useCallback(
        async (file: File) => {
            let conv = currentConversation
            if (!conv) {
                conv = await createConversation(`📸 ${file.name}`)
            }
            if (!conv) return

            setError(null)
            try {
                const cid = conv.id
                const userLine = `📸 [上传图片: ${file.name}]`
                const withUser: Conversation = {
                    ...conv,
                    messages: [...conv.messages, { text: userLine, role: 'user' as const }],
                }
                const placeholder = {
                    text: '',
                    role: 'assistant' as const,
                    meta: {
                        reasoning_trace: [] as ReasoningStep[],
                        streaming: true as const,
                    },
                }
                const withAssistant: Conversation = {
                    ...withUser,
                    messages: [...withUser.messages, placeholder],
                }
                setCurrentConversation(withAssistant)
                setConversations((prev) =>
                    prev.map((c) => (c.id === cid ? { ...c, messages: withAssistant.messages } : c))
                )

                type StreamEnd = 'none' | 'done' | 'error'
                let streamEnd: StreamEnd = 'none'
                await chatAPI.uploadImageStream(file, cid, (ev) => {
                    if (ev.type === 'step') {
                        setCurrentConversation((prev) => {
                            if (!prev || prev.id !== cid) return prev
                            const msgs = [...prev.messages]
                            const li = msgs.length - 1
                            const last = msgs[li]
                            if (!last || last.role !== 'assistant') return prev
                            msgs[li] = {
                                ...last,
                                meta: {
                                    ...last.meta,
                                    reasoning_trace: ev.reasoning_trace,
                                    streaming: true,
                                    partial_total_seconds: ev.partial_total_seconds,
                                },
                            }
                            return { ...prev, messages: msgs }
                        })
                    } else if (ev.type === 'done') {
                        streamEnd = 'done'
                        setCurrentConversation((prev) => {
                            if (!prev || prev.id !== cid) return prev
                            const msgs = [...prev.messages]
                            const li = msgs.length - 1
                            if (li < 0 || msgs[li].role !== 'assistant') return prev
                            msgs[li] = {
                                text: ev.response,
                                role: 'assistant',
                                meta: {
                                    reasoning_trace: ev.reasoning_trace,
                                    total_seconds: ev.total_seconds,
                                    cached: ev.cached,
                                    streaming: false,
                                },
                            }
                            return { ...prev, messages: msgs, id: ev.conversation_id }
                        })
                        setConversations((p) =>
                            p.map((c) => {
                                if (c.id !== cid) return c
                                const msgs = [...c.messages]
                                const li = msgs.length - 1
                                if (li < 0 || msgs[li].role !== 'assistant') return c
                                msgs[li] = {
                                    text: ev.response,
                                    role: 'assistant',
                                    meta: {
                                        reasoning_trace: ev.reasoning_trace,
                                        total_seconds: ev.total_seconds,
                                        cached: ev.cached,
                                        streaming: false,
                                    },
                                }
                                return { ...c, messages: msgs, id: ev.conversation_id, last_reply_cached: ev.cached }
                            })
                        )
                    } else if (ev.type === 'error') {
                        streamEnd = 'error'
                        setError(ev.message)
                        setCurrentConversation((prev) => {
                            if (!prev) return prev
                            return { ...prev, messages: prev.messages.slice(0, -2) }
                        })
                    }
                })

                if (streamEnd === 'none') {
                    const res = await chatAPI.uploadImage(file, cid)
                    setCurrentConversation((prev) => {
                        if (!prev || prev.id !== cid) return prev
                        const msgs = [...prev.messages]
                        const li = msgs.length - 1
                        if (li < 0 || msgs[li].role !== 'assistant') return prev
                        msgs[li] = {
                            text: res.response,
                            role: 'assistant',
                            meta: {
                                reasoning_trace: res.reasoning_trace ?? [],
                                total_seconds: res.total_seconds,
                                cached: res.cached,
                                streaming: false,
                            },
                        }
                        return { ...prev, messages: msgs, id: res.conversation_id }
                    })
                    setConversations((p) =>
                        p.map((c) => {
                            if (c.id !== cid) return c
                            const msgs = [...c.messages]
                            const li = msgs.length - 1
                            if (li < 0 || msgs[li].role !== 'assistant') return c
                            msgs[li] = {
                                text: res.response,
                                role: 'assistant',
                                meta: {
                                    reasoning_trace: res.reasoning_trace ?? [],
                                    total_seconds: res.total_seconds,
                                    cached: res.cached,
                                    streaming: false,
                                },
                            }
                            return { ...c, messages: msgs, id: res.conversation_id, last_reply_cached: !!res.cached }
                        })
                    )
                }
            } catch (err) {
                const message = err instanceof Error ? err.message : 'Failed to upload image'
                setError(message)
                console.error('Error uploading image:', err)
                try {
                    const res = await chatAPI.uploadImage(file, conv.id)
                    setCurrentConversation((prev) => {
                        if (!prev || prev.id !== conv.id) return prev
                        const msgs = [...prev.messages]
                        const li = msgs.length - 1
                        if (li >= 0 && msgs[li].role === 'assistant' && !msgs[li].text) {
                            msgs[li] = {
                                text: res.response,
                                role: 'assistant',
                                meta: {
                                    reasoning_trace: res.reasoning_trace ?? [],
                                    total_seconds: res.total_seconds,
                                    cached: res.cached,
                                    streaming: false,
                                },
                            }
                        }
                        return { ...prev, messages: msgs, id: res.conversation_id }
                    })
                    setConversations((p) =>
                        p.map((c) => {
                            if (c.id !== conv.id) return c
                            const msgs = [...c.messages]
                            const li = msgs.length - 1
                            if (li >= 0 && msgs[li].role === 'assistant' && !msgs[li].text) {
                                msgs[li] = {
                                    text: res.response,
                                    role: 'assistant',
                                    meta: {
                                        reasoning_trace: res.reasoning_trace ?? [],
                                        total_seconds: res.total_seconds,
                                        cached: res.cached,
                                        streaming: false,
                                    },
                                }
                            }
                            return { ...c, messages: msgs, id: res.conversation_id, last_reply_cached: !!res.cached }
                        })
                    )
                    setError(null)
                } catch {
                    setCurrentConversation((prev) => {
                        if (!prev) return prev
                        const msgs = [...prev.messages]
                        if (msgs.length >= 2) {
                            const a = msgs[msgs.length - 1]
                            const u = msgs[msgs.length - 2]
                            if (a?.role === 'assistant' && !a.text && u?.role === 'user') {
                                msgs.pop()
                                msgs.pop()
                            }
                        }
                        return { ...prev, messages: msgs }
                    })
                }
            }
        },
        [currentConversation, createConversation]
    )

    useEffect(() => {
        loadConversations()
    }, [])

    return (
        <ConversationContext.Provider
            value={{
                conversations,
                currentConversation,
                isLoading,
                error,
                loadConversations,
                selectConversation,
                createConversation,
                deleteConversation,
                deleteConversations,
                sendMessage,
                uploadImage,
            }}
        >
            {children}
        </ConversationContext.Provider>
    )
}

export const useConversation = () => {
    const context = useContext(ConversationContext)
    if (!context) {
        throw new Error('useConversation must be used within ConversationProvider')
    }
    return context
}

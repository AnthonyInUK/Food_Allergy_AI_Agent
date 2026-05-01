'use client'

import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Send, Paperclip, Zap, X, ImagePlus } from 'lucide-react'
import { useConversation } from '@/context/ConversationContext'

const DEFAULT_IMAGE_PROMPT =
    '请结合附图分析图中的食品，并查询过敏原、配料等相关信息。'

const MAX_TEXTAREA_PX = 200

export const ChatInput: React.FC = () => {
    const [message, setMessage] = useState('')
    const [pendingImage, setPendingImage] = useState<File | null>(null)
    const [previewUrl, setPreviewUrl] = useState<string | null>(null)
    const [isSending, setIsSending] = useState(false)
    const [isUploading, setIsUploading] = useState(false)
    const [dragActive, setDragActive] = useState(false)
    const dragDepth = useRef(0)
    const attachInputRef = useRef<HTMLInputElement>(null)
    const fastUploadInputRef = useRef<HTMLInputElement>(null)
    const textareaRef = useRef<HTMLTextAreaElement>(null)
    const { sendMessage, uploadImage, isLoading } = useConversation()

    useEffect(() => {
        if (!pendingImage) {
            setPreviewUrl(null)
            return
        }
        const u = URL.createObjectURL(pendingImage)
        setPreviewUrl(u)
        return () => URL.revokeObjectURL(u)
    }, [pendingImage])

    const inputBusy = isSending || isUploading

    const autosizeTextarea = useCallback(() => {
        const el = textareaRef.current
        if (!el) return
        el.style.height = 'auto'
        el.style.height = `${Math.min(el.scrollHeight, MAX_TEXTAREA_PX)}px`
    }, [])

    useEffect(() => {
        autosizeTextarea()
    }, [message, autosizeTextarea])

    const pickImageFile = useCallback((file: File | undefined | null) => {
        if (file && file.type.startsWith('image/')) {
            setPendingImage(file)
        }
    }, [])

    const handleSendMessage = async (e: React.FormEvent) => {
        e.preventDefault()
        const trimmed = message.trim()
        if ((!trimmed && !pendingImage) || inputBusy) return

        const textToSend = trimmed || DEFAULT_IMAGE_PROMPT
        const img = pendingImage
        setMessage('')
        setPendingImage(null)
        setIsSending(true)
        try {
            await sendMessage(textToSend, img ? { image: img } : undefined)
        } finally {
            setIsSending(false)
        }
    }

    const handleAttachChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        pickImageFile(e.target.files?.[0] ?? null)
        if (attachInputRef.current) attachInputRef.current.value = ''
    }

    const handleFastUploadChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (!file) return
        setIsUploading(true)
        try {
            await uploadImage(file)
        } finally {
            setIsUploading(false)
            if (fastUploadInputRef.current) fastUploadInputRef.current.value = ''
        }
    }

    const onPaste = (e: React.ClipboardEvent) => {
        const items = e.clipboardData?.files
        if (!items?.length) return
        const f = Array.from(items).find((x) => x.type.startsWith('image/'))
        if (f) {
            e.preventDefault()
            setPendingImage(f)
        }
    }

    const canSend =
        (message.trim().length > 0 || pendingImage !== null) &&
        !isSending &&
        !isUploading

    const onTextareaKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            if (!canSend) return
            void handleSendMessage(e as unknown as React.FormEvent)
        }
    }

    const onDragEnter = (e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        dragDepth.current += 1
        if (dragDepth.current === 1) setDragActive(true)
    }

    const onDragLeave = (e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        dragDepth.current = Math.max(0, dragDepth.current - 1)
        if (dragDepth.current === 0) setDragActive(false)
    }

    const onDragOver = (e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
    }

    const onDrop = (e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        dragDepth.current = 0
        setDragActive(false)
        const f = Array.from(e.dataTransfer.files).find((x) => x.type.startsWith('image/'))
        pickImageFile(f ?? null)
    }

    return (
        <div className="w-full flex-shrink-0 border-t border-gray-200 bg-gray-50/90 backdrop-blur-sm px-3 sm:px-4 pt-3 pb-[max(1rem,env(safe-area-inset-bottom))]">
            <form onSubmit={handleSendMessage} className="max-w-4xl mx-auto">
                <div
                    role="group"
                    aria-label="消息输入：可输入文字并拖入或粘贴图片"
                    onDragEnter={onDragEnter}
                    onDragLeave={onDragLeave}
                    onDragOver={onDragOver}
                    onDrop={onDrop}
                    onClick={() => textareaRef.current?.focus()}
                    className={[
                        'relative rounded-2xl border-2 bg-white shadow-sm transition-colors cursor-text',
                        dragActive ? 'border-blue-500 ring-2 ring-blue-200' : 'border-gray-200 hover:border-gray-300',
                    ].join(' ')}
                >
                    {dragActive && (
                        <div className="absolute inset-0 z-10 flex items-center justify-center rounded-2xl bg-blue-500/10 pointer-events-none border-2 border-dashed border-blue-500">
                            <p className="text-sm font-medium text-blue-800 flex items-center gap-2 px-4 text-center">
                                <ImagePlus className="shrink-0" size={22} />
                                松开鼠标将图片加入本条消息
                            </p>
                        </div>
                    )}

                    {previewUrl && pendingImage && (
                        <div
                            className="flex items-center gap-2 px-3 pt-3 pb-2 border-b border-gray-100"
                            onClick={(e) => e.stopPropagation()}
                        >
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img
                                src={previewUrl}
                                alt=""
                                className="h-16 w-16 rounded-xl object-cover border border-gray-200 flex-shrink-0"
                            />
                            <div className="min-w-0 flex-1">
                                <p className="text-xs font-medium text-gray-800 truncate">{pendingImage.name}</p>
                                <p className="text-[11px] text-gray-500 mt-0.5">
                                    将随文字一并发送（智能对话）
                                </p>
                            </div>
                            <button
                                type="button"
                                onClick={() => setPendingImage(null)}
                                className="p-2 rounded-full hover:bg-gray-100 text-gray-600 flex-shrink-0"
                                title="移除图片"
                            >
                                <X size={18} />
                            </button>
                        </div>
                    )}

                    <div className="flex items-end gap-1.5 sm:gap-2 px-2 py-2 sm:px-3 sm:py-2.5">
                        <div
                            className="flex flex-col gap-1 pb-0.5 flex-shrink-0"
                            onClick={(e) => e.stopPropagation()}
                        >
                            <button
                                type="button"
                                onClick={() => attachInputRef.current?.click()}
                                disabled={inputBusy}
                                className="w-9 h-9 rounded-lg text-gray-600 hover:bg-gray-100 disabled:opacity-40 flex items-center justify-center"
                                title="从相册选择图片（与文字一起发）"
                            >
                                <Paperclip size={20} strokeWidth={2} />
                            </button>
                            <button
                                type="button"
                                onClick={() => fastUploadInputRef.current?.click()}
                                disabled={inputBusy}
                                className="w-9 h-9 rounded-lg text-amber-700 hover:bg-amber-50 disabled:opacity-40 flex items-center justify-center"
                                title="仅快速查本地库（不走对话图）"
                            >
                                <Zap size={18} className={isUploading ? 'animate-pulse' : ''} />
                            </button>
                        </div>

                        <input
                            ref={attachInputRef}
                            type="file"
                            accept="image/*"
                            onChange={handleAttachChange}
                            className="hidden"
                            disabled={inputBusy}
                        />
                        <input
                            ref={fastUploadInputRef}
                            type="file"
                            accept="image/*"
                            onChange={handleFastUploadChange}
                            className="hidden"
                            disabled={inputBusy}
                        />

                        <textarea
                            ref={textareaRef}
                            value={message}
                            onChange={(e) => setMessage(e.target.value)}
                            onPaste={onPaste}
                            onKeyDown={onTextareaKeyDown}
                            onClick={(e) => e.stopPropagation()}
                            rows={1}
                            placeholder="输入消息… 可将图片拖入本框或粘贴截图，与文字一起发送（Enter 发送，Shift+Enter 换行）"
                            className="flex-1 min-w-0 min-h-[44px] max-h-[200px] py-2.5 px-1 sm:px-2 bg-transparent border-0 focus:outline-none focus:ring-0 text-gray-900 text-[15px] sm:text-base leading-relaxed resize-none overflow-y-auto"
                            disabled={inputBusy}
                            aria-label="消息文字"
                        />

                        <div className="flex-shrink-0 pb-0.5" onClick={(e) => e.stopPropagation()}>
                            <button
                                type="submit"
                                disabled={!canSend}
                                className="w-10 h-10 sm:w-11 sm:h-11 rounded-xl bg-gray-900 hover:bg-gray-800 disabled:bg-gray-300 text-white flex items-center justify-center transition-colors"
                                title="发送"
                            >
                                <Send size={18} className={isSending ? 'animate-pulse' : ''} />
                            </button>
                        </div>
                    </div>
                </div>

                <p className="text-[11px] text-gray-500 mt-2 px-1 text-center sm:text-left">
                    与 ChatGPT 类似：同一输入框内<strong className="font-medium text-gray-700"> 打字 + 拖图/贴图 </strong>
                    后点发送，走智能对话；左侧闪电为「仅本地库」快捷入口。
                    {isLoading ? ' 会话列表同步中…' : ''}
                </p>
            </form>
        </div>
    )
}

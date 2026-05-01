'use client'

import React, { useState } from 'react'
import { Plus, Trash2, MessageCircle, PanelLeftClose } from 'lucide-react'
import { useConversation } from '@/context/ConversationContext'
import { formatDistanceToNow } from 'date-fns'
import { zhCN } from 'date-fns/locale'

interface SidebarProps {
    onToggleCollapse?: () => void
    isCollapsible?: boolean
}

export const Sidebar: React.FC<SidebarProps> = ({ onToggleCollapse, isCollapsible = false }) => {
    const {
        conversations,
        currentConversation,
        selectConversation,
        createConversation,
        deleteConversation,
        deleteConversations,
        isLoading,
    } = useConversation()
    const [hoverId, setHoverId] = useState<string | null>(null)
    const [selectMode, setSelectMode] = useState(false)
    const [selectedIds, setSelectedIds] = useState<string[]>([])
    const [pendingDeleteIds, setPendingDeleteIds] = useState<string[]>([])
    const [isConfirmOpen, setIsConfirmOpen] = useState(false)

    const handleDelete = async (e: React.MouseEvent, id: string) => {
        e.stopPropagation()
        setPendingDeleteIds([id])
        setIsConfirmOpen(true)
    }

    const toggleSelected = (id: string) => {
        setSelectedIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]))
    }

    const handleBulkDelete = async () => {
        if (!selectedIds.length) return
        setPendingDeleteIds(selectedIds)
        setIsConfirmOpen(true)
    }

    const closeConfirm = () => {
        setIsConfirmOpen(false)
        setPendingDeleteIds([])
    }

    const confirmDelete = async () => {
        if (!pendingDeleteIds.length) {
            closeConfirm()
            return
        }
        if (pendingDeleteIds.length === 1) {
            await deleteConversation(pendingDeleteIds[0])
        } else {
            await deleteConversations(pendingDeleteIds)
            setSelectedIds([])
            setSelectMode(false)
        }
        closeConfirm()
    }

    return (
        <div className="w-[320px] bg-[#111420] border-r border-white/10 overflow-hidden flex flex-col h-screen relative">
            <div className="p-5 border-b border-white/10">
                <div className="mb-5 flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                        <div className="h-7 w-7 rounded-full bg-blue-500/20 text-blue-400 flex items-center justify-center">
                            <MessageCircle size={16} />
                        </div>
                        <span className="text-2xl font-semibold tracking-tight text-blue-400">FoodSeek</span>
                    </div>
                    {isCollapsible ? (
                        <button
                            onClick={() => onToggleCollapse?.()}
                            className="shrink-0 p-2 rounded-xl text-[#a6adbb] hover:bg-white/10 transition"
                            title="Collapse sidebar"
                        >
                            <PanelLeftClose size={20} />
                        </button>
                    ) : null}
                </div>
                <div className="flex items-center gap-2">
                    <button
                        onClick={() => createConversation()}
                        disabled={isLoading}
                        className="flex-1 flex items-center justify-center gap-2 bg-white/20 hover:bg-white/25 disabled:bg-white/10 text-white py-3 px-4 rounded-full transition font-semibold text-[34px]"
                    >
                        <Plus size={18} />
                        New chat
                    </button>
                    <button
                        onClick={() => {
                            if (selectMode) setSelectedIds([])
                            setSelectMode((v) => !v)
                        }}
                        disabled={isLoading || conversations.length === 0}
                        className="px-3 py-3 rounded-full bg-white/10 hover:bg-white/20 text-white text-sm disabled:opacity-50"
                        title={selectMode ? 'Cancel selection' : 'Select multiple'}
                    >
                        {selectMode ? 'Cancel' : 'Select'}
                    </button>
                    {selectMode ? (
                        <button
                            onClick={handleBulkDelete}
                            disabled={isLoading || selectedIds.length === 0}
                            className="px-3 py-3 rounded-full bg-red-500/80 hover:bg-red-500 text-white text-sm disabled:opacity-50"
                            title="Delete selected conversations"
                        >
                            Delete ({selectedIds.length})
                        </button>
                    ) : null}
                </div>
            </div>

            <div className="flex-1 overflow-y-auto">
                {conversations.length === 0 ? (
                    <div className="p-4 text-center text-[#8f97a7] text-sm">
                        <p>No conversations yet</p>
                    </div>
                ) : (
                    <div className="space-y-1 p-3">
                        {conversations.map((conv) => (
                            <div
                                key={conv.id}
                                onMouseEnter={() => setHoverId(conv.id)}
                                onMouseLeave={() => setHoverId(null)}
                                onClick={() => (selectMode ? toggleSelected(conv.id) : selectConversation(conv.id))}
                                className={`p-3 rounded-xl cursor-pointer transition-all group ${
                                    currentConversation?.id === conv.id
                                        ? 'bg-white/14 text-white'
                                        : 'hover:bg-white/10 text-[#c8cfdd]'
                                }`}
                            >
                                <div className="flex items-start justify-between gap-2">
                                    <div className="flex-1 min-w-0">
                                        <div className="flex items-center gap-2 min-w-0">
                                            {selectMode ? (
                                                <input
                                                    type="checkbox"
                                                    checked={selectedIds.includes(conv.id)}
                                                    onChange={() => toggleSelected(conv.id)}
                                                    onClick={(e) => e.stopPropagation()}
                                                    className="h-4 w-4 accent-blue-500"
                                                />
                                            ) : null}
                                            <p className="font-medium text-sm truncate flex items-center gap-2 min-w-0">
                                                <MessageCircle size={16} className="shrink-0 text-[#8e96a8]" />
                                                <span className="truncate">{conv.title || 'Untitled'}</span>
                                            </p>
                                            {conv.last_reply_cached ? (
                                                <span
                                                    className={`shrink-0 inline-flex items-center rounded px-1.5 py-0.5 text-[10px] leading-none ${
                                                        currentConversation?.id === conv.id
                                                            ? 'bg-blue-400/30 text-blue-100'
                                                            : 'bg-blue-500/20 text-blue-300'
                                                    }`}
                                                >
                                                    缓存命中
                                                </span>
                                            ) : null}
                                        </div>
                                        {conv.created_at && (
                                            <p
                                                className={`text-xs mt-1 ${
                                                    currentConversation?.id === conv.id ? 'text-[#d8def0]' : 'text-[#8f97a7]'
                                                }`}
                                            >
                                                {formatDistanceToNow(new Date(conv.created_at), {
                                                    addSuffix: true,
                                                    locale: zhCN,
                                                })}
                                            </p>
                                        )}
                                    </div>
                                    {!selectMode && hoverId === conv.id && (
                                        <button
                                            onClick={(e) => handleDelete(e, conv.id)}
                                            className={`p-1 rounded transition-all hover:scale-110 ${
                                                currentConversation?.id === conv.id
                                                    ? 'hover:bg-red-500/80 text-white'
                                                    : 'hover:bg-red-500/20 text-red-300'
                                            }`}
                                            title="Delete conversation"
                                        >
                                            <Trash2 size={16} />
                                        </button>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            <div className="p-4 border-t border-white/10 text-xs text-[#7f8798] text-center">
                <p>Food Allergy AI Agent</p>
            </div>

            {isConfirmOpen ? (
                <div
                    className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-4"
                    onClick={closeConfirm}
                >
                    <div
                        className="w-full max-w-md rounded-2xl border border-white/15 bg-[#151a28] p-5 shadow-2xl"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <h3 className="text-lg font-semibold text-white">确认删除</h3>
                        <p className="mt-2 text-sm text-[#b8c0d4]">
                            {pendingDeleteIds.length > 1
                                ? `将永久删除 ${pendingDeleteIds.length} 个对话，删除后无法恢复。`
                                : '将永久删除这条对话，删除后无法恢复。'}
                        </p>
                        <div className="mt-5 flex justify-end gap-2">
                            <button
                                onClick={closeConfirm}
                                disabled={isLoading}
                                className="rounded-lg bg-white/10 px-4 py-2 text-sm text-white hover:bg-white/20 disabled:opacity-50"
                            >
                                取消
                            </button>
                            <button
                                onClick={confirmDelete}
                                disabled={isLoading}
                                className="rounded-lg bg-red-500/90 px-4 py-2 text-sm font-medium text-white hover:bg-red-500 disabled:opacity-50"
                            >
                                {isLoading ? '删除中...' : '确认删除'}
                            </button>
                        </div>
                    </div>
                </div>
            ) : null}
        </div>
    )
}

import React, { useState, useEffect, useRef } from 'react'
import { Send, RotateCcw, Paperclip, X, ThumbsUp, ThumbsDown } from 'lucide-react'
import { api } from '../services/api'
import toast from 'react-hot-toast'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useLocation, useNavigate } from 'react-router-dom'
import Avatar3D, { avatarBus } from '../components/Avatar3D'


interface AttachmentPreview {
    name: string
    type: string
    url?: string
    size: number
}

interface Message {
    id: string
    role: 'user' | 'assistant'
    content: string
    timestamp: string
    attachments?: AttachmentPreview[],
    feedback_id?: string | null
}

const UserDashboard: React.FC = () => {
    const [messages, setMessages] = useState<Message[]>([])
    const [inputMessage, setInputMessage] = useState('')
    const [isLoading, setIsLoading] = useState(false)
    const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
    const location = useLocation()
    const navigate = useNavigate()
    const [pendingFiles, setPendingFiles] = useState<File[]>([])
    const [isDragging, setIsDragging] = useState(false)
    const fileInputRef = useRef<HTMLInputElement>(null)
    const messagesEndRef = useRef<HTMLDivElement>(null)
    const dragCounter = useRef(0)
    const [showAssistant, setShowAssistant] = useState(true)
    const [feedbackSent, setFeedbackSent] = useState<{ [messageId: string]: boolean }>({})
    const streamControllerRef = useRef<AbortController | null>(null)

    useEffect(() => {
        // Greet user on initial load
        if (messages.length === 0 && !location.search.includes('sid=')) {
            const greetingMessage: Message = {
                id: 'initial-greeting',
                role: 'assistant',
                content: 'Xin chào! Mình là trợ lý AI nội bộ của DalatHasfarm. Mình có thể hỗ trợ bạn tra cứu thông tin hoặc giải đáp công việc gì hôm nay?',
                timestamp: new Date().toISOString(),
            };
            setMessages([greetingMessage]);
            avatarBus.set('happy');
            setTimeout(() => avatarBus.set('idle'), 2500); // Return to idle after wave
        }
    }, [location.search]);

    useEffect(() => {
        const prevent = (e: DragEvent) => { e.preventDefault(); e.stopPropagation(); }
        const onEnter = (e: DragEvent) => { prevent(e); dragCounter.current += 1; setIsDragging(true) }
        const onLeave = (e: DragEvent) => { prevent(e); dragCounter.current = Math.max(0, dragCounter.current - 1); if (dragCounter.current === 0) setIsDragging(false) }
        const onOver = (e: DragEvent) => { prevent(e) }
        const onDrop = (e: DragEvent) => {
            prevent(e); setIsDragging(false); dragCounter.current = 0
            const files = (e.dataTransfer && e.dataTransfer.files) ? Array.from(e.dataTransfer.files) : []
            if (files.length > 0) setPendingFiles(prev => [...prev, ...files])
        }
        window.addEventListener('dragenter', onEnter)
        window.addEventListener('dragleave', onLeave)
        window.addEventListener('dragover', onOver)
        window.addEventListener('drop', onDrop)
        return () => {
            window.removeEventListener('dragenter', onEnter)
            window.removeEventListener('dragleave', onLeave)
            window.removeEventListener('dragover', onOver)
            window.removeEventListener('drop', onDrop)
        }
    }, [])

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }

    useEffect(() => {
        scrollToBottom()
    }, [messages])


    const sendMessage = async () => {
        if ((!inputMessage.trim() && pendingFiles.length === 0) || isLoading) return
        // Cancel any ongoing stream before starting a new one
        try { streamControllerRef.current?.abort(); } catch (_) {}
        streamControllerRef.current = null

        const previews: AttachmentPreview[] = pendingFiles.map(f => ({
            name: f.name,
            type: f.type,
            size: f.size,
            url: f.type.startsWith('image/') ? URL.createObjectURL(f) : undefined,
        }))

        const userMessage: Message = {
            id: `${Date.now()}`,
            role: 'user',
            content: inputMessage || (pendingFiles.length > 0 ? `[Đính kèm ${pendingFiles.length} tệp]` : ''),
            timestamp: new Date().toISOString(),
            attachments: previews
        }





        setMessages(prev => [...prev, userMessage])
        setInputMessage('')
        setIsLoading(true)
        avatarBus.set('thinking')

        try {
            if (pendingFiles.length > 0) {
                // Non-stream path for file uploads
                const form = new FormData()
                form.append('message', inputMessage)
                if (currentSessionId) form.append('session_id', currentSessionId)
                for (const f of pendingFiles) form.append('files', f)
                const response = await api.post('/chat/send-with-files', form, { headers: { 'Content-Type': 'multipart/form-data' } })

                const aiMessage: Message = {
                    id: `${response.data.id}-a`,
                    role: 'assistant',
                    content: response.data.response,
                    timestamp: new Date().toISOString()
                }
                setMessages(prev => [...prev, aiMessage])
                avatarBus.set('talk')
                setTimeout(() => avatarBus.set('idle'), 1200)
                const newSid = response.data.session_id as string
                if (newSid && newSid !== currentSessionId) {
                    const params = new URLSearchParams(location.search)
                    params.set('sid', newSid)
                    const search = `?${params.toString()}`
                    navigate({ pathname: location.pathname, search }, { replace: true })
                    setCurrentSessionId(newSid)
                    window.dispatchEvent(new Event('chat:sessions:refresh'))
                }
                setPendingFiles([])
            } else {
                // Streaming path for normal text chat
                const assistantId = `${Date.now() + 1}`
                const placeholder: Message = {
                    id: assistantId,
                    role: 'assistant',
                    content: '',
                    timestamp: new Date().toISOString()
                }
                setMessages(prev => [...prev, placeholder])

                const token = localStorage.getItem('token')
                const controller = new AbortController()
                streamControllerRef.current = controller
                const res = await fetch('/api/chat/stream', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        ...(token ? { Authorization: `Bearer ${token}` } : {}),
                    },
                    body: JSON.stringify({ message: inputMessage, session_id: currentSessionId }),
                    signal: controller.signal,
                })
                if (!res.ok || !res.body) throw new Error('Stream failed')

                const reader = res.body.getReader()
                const decoder = new TextDecoder()
                let acc = ''
                let gotSid = false
                let streamSid: string | null = null

                while (true) {
                    const { value, done } = await reader.read()
                    if (done) break
                    const chunk = decoder.decode(value, { stream: true })
                    acc += chunk

                    if (!gotSid) {
                        const m = acc.match(/<<SID:([a-f0-9-]+)>>/i)
                        if (m) {
                            const sid = m[1]
                            acc = acc.replace(m[0], '')
                            gotSid = true
                            streamSid = sid
                            if (sid && sid !== currentSessionId) {
                                const params = new URLSearchParams(location.search)
                                params.set('sid', sid)
                                const search = `?${params.toString()}`
                                navigate({ pathname: location.pathname, search }, { replace: true })
                                setCurrentSessionId(sid)
                                window.dispatchEvent(new Event('chat:sessions:refresh'))
                            }
                        }
                    }

                    setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content: acc } : m))
                }

                avatarBus.set('talk')
                setTimeout(() => avatarBus.set('idle'), 1200)

                // Refresh messages to get real IDs from server for feedback
                try {
                    const sidToLoad = streamSid || currentSessionId
                    if (sidToLoad) {
                        const response = await api.get(`/chat/sessions/${sidToLoad}/messages`)
                        const serverMessages: Message[] = []
                        const initialFeedback: { [messageId: string]: boolean } = {}
                        ;(response.data || []).forEach((m: any) => {
                            if (m.message) {
                                serverMessages.push({ id: `${m.id}-u`, role: 'user', content: m.message, timestamp: m.timestamp })
                            }
                            if (m.response) {
                                const assistantMessageId = `${m.id}-a`
                                serverMessages.push({ id: assistantMessageId, role: 'assistant', content: m.response, timestamp: m.timestamp })
                                if (m.feedback_rating === 1 || m.feedback_rating === -1) {
                                    initialFeedback[assistantMessageId] = true
                                }
                            }
                        })
                        setMessages(serverMessages)
                        setFeedbackSent(prev => ({ ...prev, ...initialFeedback }))
                    }
                } catch (_) {}

                setPendingFiles([])
                return
            }
        } catch (error: any) {
            if (error?.name === 'AbortError') {
                // Stream cancelled by user (new message/reset)
                return
            }
            avatarBus.set('sad')
            toast.error(error.response?.data?.detail || 'Có lỗi xảy ra khi gửi tin nhắn')
        } finally {
            setIsLoading(false)
            streamControllerRef.current = null
        }
    }

    // Load session when URL changes
    useEffect(() => {
        const params = new URLSearchParams(location.search)
        const sid = params.get('sid')
        if (sid && sid !== currentSessionId) {
            (async () => {
                try {
                    const response = await api.get(`/chat/sessions/${sid}/messages`)
                    const serverMessages: Message[] = [];
                    const initialFeedback: { [messageId: string]: boolean } = {};
                    (response.data || []).forEach((m: any) => {
                        if (m.message) {
                            serverMessages.push({ id: `${m.id}-u`, role: 'user', content: m.message, timestamp: m.timestamp });
                        }
                        if (m.response) {
                            const assistantMessageId = `${m.id}-a`;
                            serverMessages.push({ id: assistantMessageId, role: 'assistant', content: m.response, timestamp: m.timestamp });
                            if (m.feedback_rating === 1 || m.feedback_rating === -1) {
                                initialFeedback[assistantMessageId] = true;
                            }
                        }
                    });
                    setMessages(serverMessages);
                    setFeedbackSent(initialFeedback);
                    setCurrentSessionId(sid);
                } catch (error) {
                    toast.error('Không thể tải lịch sử chat')
                }
            })()
        } else if (!sid && currentSessionId) {
            // Clear when no sid in URL
            setMessages([])
            setCurrentSessionId(null)
        }
    }, [location.search, currentSessionId])

    const resetChat = () => {
        try { streamControllerRef.current?.abort(); } catch (_) {}
        streamControllerRef.current = null
        setMessages([])
        setCurrentSessionId(null)
        setFeedbackSent({})
        navigate('/dashboard', { replace: true })
    }

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            sendMessage()
        }
    }

    const handleFeedback = async (messageId: string, rating: 1 | -1) => {
        const originalId = messageId.replace(/-a$/, '') // Strip suffix for API
        try {
            await api.post('/feedback', { chat_message_id: originalId, rating })
            setFeedbackSent(prev => ({ ...prev, [messageId]: true }))
            toast.success('Cảm ơn bạn đã đánh giá!')
            if (rating === 1) {
                avatarBus.set('yes')
            } else {
                avatarBus.set('no')
            }
            setTimeout(() => avatarBus.set('idle'), 2500)
        } catch (error) {
            toast.error('Gửi đánh giá thất bại.')
        }
    }

    useEffect(() => {
        return () => { try { streamControllerRef.current?.abort(); } catch (_) {} }
    }, [])

    return (
        <div className="h-full flex flex-col">
            {/* Header */}
            <div className="bg-white border-b px-3 sm:px-4 py-2 dark:bg-[#0d0d0d] dark:border-gray-800">
                <div className="flex items-center justify-between">
                    <h1 className="text-lg sm:text-xl font-semibold text-gray-900 dark:text-gray-100">Chat AI</h1>
                    <div className="flex items-center gap-2">
                        <style>{`
                .no-scrollbar::-webkit-scrollbar { display: none; }
                .no-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
            `}</style>

                        <button
                            onClick={() => setShowAssistant(v => !v)}
                            className="btn-secondary-sm hidden lg:inline-flex items-center gap-1.5"
                            title={showAssistant ? 'Ẩn Trợ lý 3D' : 'Hiện Trợ lý 3D'}
                        >
                            <span className="text-sm">{showAssistant ? 'Ẩn Trợ lý 3D' : 'Hiện Trợ lý 3D'}</span>
                        </button>

                        <button
                            onClick={resetChat}
                            className="btn-secondary-sm flex items-center gap-1.5"
                            title="Bắt đầu cuộc trò chuyện mới"
                        >
                            <RotateCcw className="h-4 w-4" />
                            <span className="text-sm">Reset</span>
                        </button>
                    </div>
                </div>
            </div>

            <div className="flex-1 min-h-0 flex gap-2 sm:gap-3">
                {/* Avatar Panel (left) */}
                {showAssistant && (
                    <div className="hidden lg:block order-2 w-[280px] xl:w-[340px] border-l bg-white dark:bg-[#0d0d0d] dark:border-gray-800">
                        <div className="sticky top-0 p-3">
                            <div className="px-1 pb-2 text-[11px] font-medium tracking-wide text-gray-500 dark:text-gray-300">Agent</div>
                            <div className="h-[340px] xl:h-[420px] w-full overflow-hidden rounded-2xl border border-white/60 bg-white/70 backdrop-blur-sm shadow-sm dark:bg-white/[0.04] dark:border-white/[0.08]">
                                <Avatar3D modelUrl="/models/AnimatedCharacter.glb" />
                            </div>
                        </div>
                    </div>
                )}
                {/* Chat Area */}
                <div className="flex-1 min-h-0 flex flex-col order-1">
                    {/* Messages */}
                    <div className="flex-1 min-h-0 overflow-y-auto no-scrollbar p-3 sm:p-4 space-y-2 bg-gradient-to-b from-white to-primary-50 dark:from-[#0a0a0a] dark:to-[#0a0a0a]" style={{ scrollbarGutter: 'stable' }}>
                        {messages.length === 0 && (
                            <div className="flex items-center justify-center h-full">
                                <div className="w-full max-w-3xl mx-auto px-4">
                                    <div className="text-center mb-3 dark:text-gray-200">
                                        <h2 className="text-2xl font-semibold mb-1.5">Bạn đang tìm kiếm về cái gì?</h2>
                                        <p className="text-sm text-gray-500 dark:text-gray-400">Hỏi bất cứ điều gì và AI sẽ trả lời dựa trên kiến thức có sẵn</p>
                                    </div>

                                    {/* Attached files preview (centered) */}
                                    {pendingFiles.length > 0 && (
                                        <div className="mb-2 flex flex-wrap gap-1.5 justify-center">
                                            {pendingFiles.map((f, idx) => (
                                                <span key={idx} className="inline-flex items-center gap-1.5 text-[11px] px-2.5 py-1.5 rounded-full bg-gray-100 dark:bg-zinc-800 ring-1 ring-gray-200 dark:ring-zinc-700">
                                                    {f.name}
                                                    <button onClick={() => setPendingFiles(pendingFiles.filter((_, i) => i !== idx))} className="opacity-70 hover:opacity-100">
                                                        <X className="h-3 w-3" />
                                                    </button>
                                                </span>
                                            ))}
                                        </div>
                                    )}

                                    {/* Drag overlay */}
                                    {isDragging && (
                                        <div className="absolute inset-0 z-20 rounded-2xl border-2 border-dashed border-[#80DB97] bg-[#80DB97]/10 flex items-center justify-center text-sm text-[#80DB97]">
                                            Thả tệp vào đây để đính kèm
                                        </div>
                                    )}

                                    {/* Center input */}
                                    <div className="relative">
                                        <div className="bg-white dark:bg-[#0f0f0f] rounded-2xl shadow-lg ring-1 ring-gray-200 dark:ring-gray-800 px-3 py-2 pr-2 flex items-center gap-2 focus-within:ring-2 focus-within:ring-[#80DB97] focus-within:shadow-[0_0_0_6px_rgba(128,219,151,0.15)]">
                                            <button onClick={() => fileInputRef.current?.click()} className="text-gray-600 hover:text-gray-900 dark:text-gray-300 dark:hover:text-white px-2">
                                                <Paperclip className="h-4 w-4" />
                                            </button>
                                            <input ref={fileInputRef} type="file" className="hidden" multiple accept=".pdf,.txt,.doc,.docx,image/*" onChange={(e) => setPendingFiles([...(pendingFiles || []), ...(Array.from(e.target.files || []))])} />
                                            <textarea
                                                value={inputMessage}
                                                onChange={(e) => setInputMessage(e.target.value)}
                                                onKeyDown={handleKeyPress}
                                                placeholder="Nhập tin nhắn của bạn..."
                                                className="flex-1 bg-transparent outline-none resize-none text-[14px] leading-5 max-h-40 pr-2 dark:text-gray-100 dark:placeholder-gray-400"
                                                rows={1}
                                                disabled={isLoading}
                                            />
                                            <button
                                                onClick={sendMessage}
                                                disabled={(inputMessage.trim().length === 0 && pendingFiles.length === 0) || isLoading}
                                                className="btn-send btn-send-sm disabled:opacity-50 disabled:cursor-not-allowed shadow-md ml-auto"
                                            >
                                                <Send className="h-4 w-4" />
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {messages.map((message) => (
                            <div
                                key={message.id}
                                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                            >
                                <div
                                    className={`${message.role === 'user' ? 'bubble-user' : 'bubble-ai'} w-fit max-w-[85%]`}
                                >
                                    <ReactMarkdown
                                        remarkPlugins={[remarkGfm]}
                                        className="prose prose-sm max-w-none break-words prose-p:my-1 prose-headings:my-1 prose-ul:my-1 prose-ol:my-1 prose-li:my-0.5 prose-pre:my-2"
                                    >
                                        {message.content}
                                    </ReactMarkdown>
                                    {message.attachments && message.attachments.length > 0 && (
                                        <div className="mt-1.5 space-y-1.5">
                                            {message.attachments.map((att, idx) => (
                                                att.url && att.type?.startsWith('image/') ? (
                                                    <img key={idx} src={att.url} alt={att.name} className="max-w-xs rounded-lg border dark:border-zinc-800" />
                                                ) : (
                                                    <div key={idx} className="text-xs text-gray-500 dark:text-gray-400">{att.name}</div>
                                                )
                                            ))}
                                        </div>
                                    )}
                                    <div className={`text-[11px] mt-0.5 ${message.role === 'user' ? 'text-primary-100' : 'text-gray-500'}`}>
                                        {new Date(message.timestamp).toLocaleTimeString('vi-VN', { timeZone: 'Asia/Ho_Chi_Minh' })}
                                    </div>
                                    {message.role === 'assistant' && (
                                        <div className="mt-1 pt-1 border-t border-white/10 flex items-center gap-1.5">
                                            {feedbackSent[message.id] ? (
                                                <span className="text-xs text-gray-400">Đã cảm ơn!</span>
                                            ) : (
                                                <>
                                                    <button onClick={() => handleFeedback(message.id, 1)} className="p-1 rounded-full hover:bg-white/20 text-gray-400 hover:text-white">
                                                        <ThumbsUp className="h-4 w-4" />
                                                    </button>
                                                    <button onClick={() => handleFeedback(message.id, -1)} className="p-1 rounded-full hover:bg-white/20 text-gray-400 hover:text-white">
                                                        <ThumbsDown className="h-4 w-4" />
                                                    </button>
                                                </>
                                            )}
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))}

                        {isLoading && (
                            <div className="flex justify-start text-sm">
                                <div className="bg-gray-100 rounded-lg px-3 py-1.5 dark:bg-[#111111]">
                                    <div className="flex items-center space-x-2">
                                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600"></div>
                                        <span className="text-gray-600 dark:text-gray-300">AI đang suy nghĩ...</span>
                                    </div>
                                </div>
                            </div>
                        )}

                        <div ref={messagesEndRef} />
                    </div>

                    {/* Input Area */}
                    {messages.length > 0 && (
                        <div className="border-t bg-white p-3 dark:bg-[#0d0d0d] dark:border-gray-800">
                            <div className="relative max-w-5xl mx-auto">
                                {/* Attached files preview */}
                                {pendingFiles.length > 0 && (
                                    <div className="mb-2 flex flex-wrap gap-2">
                                        {pendingFiles.map((f, idx) => (
                                            <span key={idx} className="inline-flex items-center gap-2 text-xs px-2 py-1 rounded-full bg-gray-100 dark:bg-zinc-800 ring-1 ring-gray-200 dark:ring-zinc-700">
                                                {f.name}
                                                <button onClick={() => setPendingFiles(pendingFiles.filter((_, i) => i !== idx))} className="opacity-70 hover:opacity-100">
                                                    <X className="h-3 w-3" />
                                                </button>
                                            </span>
                                        ))}
                                    </div>
                                )}
                                {/* Drag overlay */}
                                {isDragging && (
                                    <div className="absolute inset-0 z-20 rounded-2xl border-2 border-dashed border-[#80DB97] bg-[#80DB97]/10 flex items-center justify-center text-sm text-[#80DB97]">
                                        Thả tệp vào đây để đính kèm
                                    </div>
                                )}
                                <div
                                    className="bg-gray-100/90 rounded-full shadow-sm ring-1 ring-gray-200 px-3 py-2 pr-2 flex items-center gap-2 dark:bg-[#0f0f0f] dark:ring-gray-800 focus-within:ring-2 focus-within:ring-[#80DB97] focus-within:shadow-[0_0_0_6px_rgba(128,219,151,0.15)]"
                                >
                                    <button onClick={() => fileInputRef.current?.click()} className="text-gray-600 hover:text-gray-900 dark:text-gray-300 dark:hover:text-white px-2">
                                        <Paperclip className="h-4 w-4" />
                                    </button>
                                    <input ref={fileInputRef} type="file" className="hidden" multiple accept=".pdf,.txt,.doc,.docx,image/*" onChange={(e) => setPendingFiles([...(pendingFiles || []), ...(Array.from(e.target.files || []))])} />
                                    <textarea
                                        value={inputMessage}
                                        onChange={(e) => setInputMessage(e.target.value)}
                                        onKeyDown={handleKeyPress}
                                        placeholder="Nhập tin nhắn của bạn..."
                                        className="flex-1 bg-transparent outline-none resize-none text-[14px] leading-5 max-h-40 pr-2 dark:text-gray-100 dark:placeholder-gray-400"
                                        rows={1}
                                        disabled={isLoading}
                                    />
                                    <button
                                        onClick={sendMessage}
                                        disabled={(inputMessage.trim().length === 0 && pendingFiles.length === 0) || isLoading}
                                        className="btn-send btn-send-sm disabled:opacity-50 disabled:cursor-not-allowed shadow-md ml-auto"
                                    >
                                        <Send className="h-4 w-4" />
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>

        </div>
    )
}

export default UserDashboard
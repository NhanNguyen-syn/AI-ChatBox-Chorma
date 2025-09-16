import React, { useEffect, useMemo, useState } from 'react'
import { api } from '../services/api'
import toast from 'react-hot-toast'
import { Trash2, RefreshCcw, MessageSquare, AlertTriangle, X } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface Session {
  id: string
  title: string
  created_at: string
  updated_at: string
}

interface ChatMessageItem {
  id: string
  message: string
  response: string
  timestamp: string
  session_id: string
}

const History: React.FC = () => {
  const [sessions, setSessions] = useState<Session[]>([])
  const [selected, setSelected] = useState<Session | null>(null)
  const [messages, setMessages] = useState<ChatMessageItem[]>([])
  const [loading, setLoading] = useState<boolean>(true)
  const [loadingMsgs, setLoadingMsgs] = useState<boolean>(false)
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [pendingId, setPendingId] = useState<string | null>(null)
  const [deleting, setDeleting] = useState(false)

  const openConfirm = (id: string) => {
    setPendingId(id)
    setConfirmOpen(true)
  }

  const closeConfirm = () => {
    if (deleting) return
    setConfirmOpen(false)
    setPendingId(null)
  }

  const confirmDelete = async () => {
    if (!pendingId) return
    try {
      setDeleting(true)
      await deleteSession(pendingId)
    } finally {
      setDeleting(false)
      setConfirmOpen(false)
      setPendingId(null)
    }
  }


  const loadSessions = async () => {
    setLoading(true)
    try {
      const res = await api.get('/chat/sessions')
      const list = res.data || []
      setSessions(list)
      // Keep current selection if still exists; otherwise pick first
      if (selected) {
        const still = list.find((x: any) => x.id === selected.id)
        if (still) {
          // refresh messages for the same session
          await pickSession(still)
        } else if (list.length > 0) {
          await pickSession(list[0])
        } else {
          setSelected(null)
          setMessages([])
        }
      } else if (list.length > 0) {
        await pickSession(list[0])
      }
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || 'Không thể tải lịch sử chat')
    } finally {
      setLoading(false)
    }
  }

  const deleteSession = async (sessionId: string) => {
    try {
      await api.delete(`/chat/sessions/${sessionId}`)
      toast.success('Đã xóa phiên chat')
      // Nếu đang xem phiên này thì clear panel bên phải
      if (selected?.id === sessionId) {
        setSelected(null)
        setMessages([])
      }
      // Cập nhật danh sách
      setSessions(prev => prev.filter(s => s.id !== sessionId))
      // Thông báo cho các khu vực khác (sidebar/chat) làm mới
      window.dispatchEvent(new Event('chat:sessions:refresh'))
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || 'Không thể xóa phiên chat')
    }
  }

  const pickSession = async (s: Session) => {
    setSelected(s)
    setLoadingMsgs(true)
    try {
      const res = await api.get(`/chat/sessions/${s.id}/messages`)
      setMessages(res.data || [])
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || 'Không thể tải tin nhắn')
    } finally {
      setLoadingMsgs(false)
    }
  }

  useEffect(() => {
    loadSessions()
  }, [])

  // Refresh when other parts (sidebar/chat) broadcast changes
  useEffect(() => {
    const h = () => loadSessions()
    window.addEventListener('chat:sessions:refresh', h)
    return () => window.removeEventListener('chat:sessions:refresh', h)
  }, [])

  const [query, setQuery] = useState('')
  const filtered = useMemo(() => {
    if (!query.trim()) return sessions
    const q = query.toLowerCase()
    return sessions.filter(s => (s.title || 'Phiên chat').toLowerCase().includes(q))
  }, [sessions, query])

  return (
    <>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Sessions list */}
        <div className="lg:col-span-1 bg-white dark:bg-zinc-900 rounded-xl shadow p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-semibold flex items-center gap-2"><MessageSquare className="h-5 w-5 text-primary-600" /> Lịch sử chat</h2>
            <button
              onClick={loadSessions}
              title="Làm mới"
              className="inline-flex items-center gap-1 text-sm text-primary-700 px-2 py-1 rounded hover:bg-primary-50 dark:hover:bg-zinc-800"
            >
              <RefreshCcw className="h-4 w-4" /> Làm mới
            </button>
          </div>

          <div className="mb-3">
            <input
              placeholder="Tìm kiếm phiên..."
              className="w-full px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 dark:bg-zinc-800 dark:border-zinc-700"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>

          {loading ? (
            <div className="h-40 flex items-center justify-center text-gray-500">Đang tải...</div>
          ) : filtered.length === 0 ? (
            <div className="text-gray-500 text-sm">Không tìm thấy phiên phù hợp</div>
          ) : (
            <ul className="divide-y divide-gray-100 dark:divide-zinc-800 max-h-[calc(100vh-240px)] overflow-y-auto pr-1" style={{ scrollbarGutter: 'stable' }}>
              {filtered.map((s) => (
                <li key={s.id} className={`py-3 ${selected?.id === s.id ? 'bg-primary-50 dark:bg-primary-500/10 rounded-md' : ''}`}>
                  <div className="flex items-center gap-3">
                    <div className="flex-1 min-w-0 cursor-pointer" onClick={() => pickSession(s)}>
                      <div className="font-medium text-gray-800 dark:text-gray-100 truncate" title={s.title}>{s.title || 'Phiên chat'}</div>
                      <div className="text-xs text-gray-500">Cập nhật: {new Date(s.updated_at).toLocaleString()}</div>
                    </div>
                    <button
                      onClick={() => openConfirm(s.id)}
                      title="Xóa phiên"
                      className="inline-flex items-center gap-1 text-red-600 hover:text-white border border-red-200 hover:bg-red-500 px-2 py-1 rounded transition-colors"
                    >
                      <Trash2 className="h-4 w-4" />
                      <span className="text-xs">Xóa</span>
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Messages */}
        <div className="lg:col-span-2 bg-white dark:bg-zinc-900 rounded-xl shadow p-4">
          <h2 className="text-lg font-semibold mb-3">Nội dung</h2>
          {!selected ? (
            <div className="h-40 flex items-center justify-center text-gray-500">Chọn một phiên chat để xem</div>
          ) : loadingMsgs ? (
            <div className="h-40 flex items-center justify-center text-gray-500">Đang tải tin nhắn...</div>
          ) : messages.length === 0 ? (
            <div className="text-gray-500 text-sm">Không có tin nhắn</div>
          ) : (
            <div className="space-y-4 max-h-[calc(100vh-240px)] overflow-y-auto pr-1" style={{ scrollbarGutter: 'stable' }}>
              {messages.map((m) => (
                <div key={m.id} className="space-y-2">
                  {m.message && (
                    <div className="flex justify-end">
                      <div className="max-w-[80%] bg-primary-600 text-white px-4 py-2 rounded-2xl shadow prose prose-invert prose-sm">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.message}</ReactMarkdown>
                      </div>
                    </div>
                  )}
                  {m.response && (
                    <div className="flex justify-start">
                      <div className="max-w-[80%] bg-gray-100 text-gray-800 px-4 py-2 rounded-2xl shadow dark:bg-zinc-800 dark:text-gray-100 prose prose-sm dark:prose-invert">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.response}</ReactMarkdown>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Confirm Delete Modal (outside grid) */}
      {confirmOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/30" onClick={closeConfirm} />
          <div className="relative bg-white dark:bg-zinc-900 rounded-lg shadow-xl w-full max-w-md mx-4 p-6">
            <button className="absolute right-3 top-3 text-gray-400 hover:text-gray-600" onClick={closeConfirm}>
              <X className="h-4 w-4" />
            </button>
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-full bg-red-100 flex items-center justify-center text-red-600">
                <AlertTriangle className="h-5 w-5" />
              </div>
              <div>
                <h3 className="text-lg font-semibold">Xóa phiên chat?</h3>
                <p className="text-sm text-gray-600">Hành động này sẽ xóa toàn bộ tin nhắn trong phiên đã chọn và không thể hoàn tác.</p>
              </div>
            </div>
            <div className="mt-5 flex justify-end gap-3">
              <button onClick={closeConfirm} disabled={deleting} className="px-4 py-2 rounded border bg-white text-gray-700 hover:bg-gray-50 disabled:opacity-50">Hủy</button>
              <button onClick={confirmDelete} disabled={deleting} className="px-4 py-2 rounded bg-red-600 text-white hover:bg-red-700 disabled:opacity-50">
                {deleting ? 'Đang xóa...' : 'Xóa'}
              </button>
            </div>
          </div>
        </div>
      )}

    </>
  )
}

export default History


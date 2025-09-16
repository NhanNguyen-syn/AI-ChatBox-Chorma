import React, { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../services/api'
import toast from 'react-hot-toast'
import { UploadCloud, FileText, RefreshCcw, Search } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'

interface DocItem {
  id: string
  filename: string
  file_type: string
  uploaded_by: string
  uploaded_at: string
  file_size: number
}

const Upload: React.FC = () => {
  const { user } = useAuth()
  const [docs, setDocs] = useState<DocItem[]>([])
  const [uploading, setUploading] = useState(false)
  const [query, setQuery] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  const loadDocs = async () => {
    try {
      const res = await api.get('/files/documents')
      setDocs(res.data || [])
    } catch (e: any) {
      // Nếu user không phải admin, API trả 403 -> hiển thị thông báo thân thiện
      if (e?.response?.status === 403) {
        toast.error('Chỉ Admin mới có quyền truy cập trang Tải lên')
      } else {
        toast.error(e?.response?.data?.detail || 'Không thể tải danh sách tài liệu')
      }
    }
  }

  useEffect(() => {
    loadDocs()
  }, [])

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return
    setUploading(true)
    let ok = 0
    for (const f of Array.from(files)) {
      const form = new FormData()
      form.append('file', f)
      try {
        await api.post('/files/upload', form, { headers: { 'Content-Type': 'multipart/form-data' } })
        ok += 1
      } catch (e: any) {
        toast.error(`${f.name}: ${e?.response?.data?.detail || 'Upload thất bại'}`)
      }
    }
    if (ok > 0) toast.success(`Tải lên thành công ${ok} tệp`)
    await loadDocs()
    setUploading(false)
  }

  const onDrop: React.DragEventHandler<HTMLDivElement> = (e) => {
    e.preventDefault(); e.stopPropagation()
    handleFiles(e.dataTransfer.files)
  }

  const filtered = useMemo(() => {
    if (!query.trim()) return docs
    const q = query.toLowerCase()
    return docs.filter(d => d.filename.toLowerCase().includes(q))
  }, [docs, query])

  return (
    <div className="space-y-6">
      <div className="bg-white dark:bg-zinc-900 rounded-xl shadow p-6">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Tải lên tài liệu</h2>
          <button onClick={loadDocs} className="inline-flex items-center gap-1 text-sm text-primary-700 hover:underline"><RefreshCcw className="h-4 w-4"/> Làm mới</button>
        </div>
        {/* Nếu không phải Admin -> hiển thị thông báo */}
        {!user?.is_admin ? (
          <div className="mt-6 text-sm text-amber-600 bg-amber-50 border border-amber-200 rounded-lg p-4">
            Tính năng tải lên chỉ dành cho Admin. Vui lòng liên hệ quản trị để cấp quyền.
          </div>
        ) : (
          <div
            className="mt-4 border-2 border-dashed rounded-2xl p-10 text-center hover:bg-gray-50 dark:hover:bg-zinc-800"
            onDragOver={(e) => { e.preventDefault(); e.stopPropagation() }}
            onDrop={onDrop}
          >
            <div className="flex flex-col items-center gap-3">
              <UploadCloud className="h-10 w-10 text-primary-600" />
              <p className="text-gray-700 dark:text-gray-300">Kéo thả tệp vào đây hoặc</p>
              <div>
                <button onClick={() => inputRef.current?.click()} className="px-4 py-2 rounded-lg bg-primary-600 text-white" disabled={uploading}>
                  {uploading ? 'Đang tải...' : 'Chọn tệp'}
                </button>
                <input ref={inputRef} id="fileInput" type="file" className="hidden" multiple accept=".pdf,.txt" onChange={(e) => handleFiles(e.target.files)} />
              </div>
              <p className="text-xs text-gray-500">Hỗ trợ: PDF, TXT (chỉ Admin)</p>
            </div>
          </div>
        )}
      </div>

      <div className="bg-white dark:bg-zinc-900 rounded-xl shadow p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Danh sách tài liệu</h2>
          <div className="relative">
            <Search className="h-4 w-4 absolute left-2 top-2.5 text-gray-400"/>
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Tìm theo tên tệp..." className="pl-8 pr-3 py-2 rounded-lg border text-sm dark:bg-zinc-800 dark:border-zinc-700"/>
          </div>
        </div>
        {filtered.length === 0 ? (
          <div className="text-gray-500 text-sm">Chưa có tài liệu</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="text-left text-gray-500">
                  <th className="py-2 pr-4">Tên tệp</th>
                  <th className="py-2 pr-4">Loại</th>
                  <th className="py-2 pr-4">Người tải lên</th>
                  <th className="py-2 pr-4">Thời gian</th>
                  <th className="py-2 pr-4">Kích thước</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((d) => (
                  <tr key={d.id} className="border-t">
                    <td className="py-2 pr-4 flex items-center gap-2"><FileText className="h-4 w-4"/> {d.filename}</td>
                    <td className="py-2 pr-4">{d.file_type.toUpperCase()}</td>
                    <td className="py-2 pr-4">{d.uploaded_by}</td>
                    <td className="py-2 pr-4">{new Date(d.uploaded_at).toLocaleString()}</td>
                    <td className="py-2 pr-4">{(d.file_size / 1024).toFixed(1)} KB</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

export default Upload

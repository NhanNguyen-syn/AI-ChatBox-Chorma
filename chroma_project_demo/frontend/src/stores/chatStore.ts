import { create } from 'zustand'
import { api } from '../services/api'
import toast from 'react-hot-toast'

export interface Session {
  id: string
  title: string
  created_at: string
  updated_at: string
}

const SESSIONS_PER_PAGE = 15;

interface ChatState {
  sessions: Session[];
  sessionPage: number;
  sessionTotal: number;
  fetchSessions: () => Promise<void>;
  setSessionPage: (page: number) => void;
  deleteSession: (sessionId: string) => Promise<void>;
}

export const useChatStore = create<ChatState>((set, get) => ({
  sessions: [],
  sessionPage: 1,
  sessionTotal: 0,

  setSessionPage: (page: number) => {
    set({ sessionPage: page });
    get().fetchSessions();
  },

  fetchSessions: async () => {
    try {
      const { sessionPage } = get();
      const res = await api.get(`/chat/sessions`, {
        params: { page: sessionPage, limit: SESSIONS_PER_PAGE }
      });
      const data = res.data;
      // Backend may return either an array (legacy) or an object { sessions, total }
      let sessions: Session[] = [];
      let total = 0;
      if (Array.isArray(data)) {
        sessions = data as Session[];
        total = sessions.length;
      } else if (data && Array.isArray(data.sessions)) {
        sessions = data.sessions as Session[];
        total = Number(data.total) || sessions.length;
      }
      set({ sessions, sessionTotal: total });
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || 'Không thể tải lịch sử chat');
      set({ sessions: [], sessionTotal: 0 });
    }
  },

  deleteSession: async (sessionId: string) => {
    try {
      await api.delete(`/chat/sessions/${sessionId}`);
      toast.success('Đã xóa phiên chat');

      const { sessions, sessionPage } = get();
      // If the deleted session was the last one on the current page, go back one page.
      if (sessions.length === 1 && sessionPage > 1) {
        set({ sessionPage: sessionPage - 1 });
      }
      // Refetch the (potentially new) current page.
      get().fetchSessions();

    } catch (e: any) {
      toast.error(e?.response?.data?.detail || 'Không thể xóa phiên chat');
    }
  },
}));


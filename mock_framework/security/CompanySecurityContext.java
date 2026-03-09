package mock_framework.security;

/**
 * 사내 보안 컨텍스트 홀더.
 * 현재 HTTP Request를 보낸 임직원의 세션 정보를 ThreadLocal로 관리합니다.
 */
public class CompanySecurityContext {

    /**
     * 현재 로그인한 임직원의 사번(Employee ID)을 조회합니다.
     * 비즈니스 로직(Service)에서 등록자/수정자 ID를 세팅할 때 반드시 이 메서드를 사용하세요.
     * * @return 8자리 사번 문자열 (예: "20260306")
     */
    public static String getCurrentEmpId() {
        // ThreadLocal에서 사번 추출하는 가상 로직
        return "20260306"; 
    }
}